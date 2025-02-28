import torch
from torch import nn
import torch.distributed
import torch.nn.functional as F
import os
from data import MathDataset
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from common import seq_len, vocab_size
import sys

dropout = 0.25
dim = 1024
batch_size = 32


def save_checkpoint(model, optimizer, epoch, step, loss, filename):
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, filename)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=16):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.to_qkv = nn.Linear(dim, dim_head * 3 * heads, bias=False)
        self.to_out = nn.Linear(dim_head * heads, dim)
        # self.register_buffer('mask', torch.tril(torch.ones(seq_len, seq_len, device=device)))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(
            3, dim=-1
        )  # shape is now (batch_size, seq_len, dim_head * heads) for q,k,v
        q = q.view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.to_out(out)


class FFN(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 3),
            nn.GELU(),
            nn.Linear(dim * 3, dim * 3),
            nn.GELU(),
            nn.Linear(dim * 3, num_experts),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class FFNNormal(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        return self.net(x)


class MoE(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.k = 2
        self.gating_function = FFN(dim, num_experts)
        scaling_factor = 4
        self.expert_weights_1 = nn.Parameter(
            torch.empty(self.num_experts, dim * scaling_factor, dim)
        )
        self.expert_weights_2 = nn.Parameter(
            torch.empty(self.num_experts, dim * scaling_factor, dim)
        )

        self.expert_biases_1 = nn.Parameter(torch.zeros(self.num_experts, dim, 1))
        self.expert_biases_2 = nn.Parameter(torch.zeros(self.num_experts, dim, 1))

        nn.init.xavier_normal_(self.expert_weights_1)
        nn.init.xavier_normal_(self.expert_weights_2)

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape
        gate_output: torch.Tensor = self.gating_function(x)

        values, indices = gate_output.topk(self.k)
        values = values / torch.sum(values, dim=-1, keepdim=True)

        ffn_weights_1 = self.expert_weights_1[indices]
        ffn_weights_2 = self.expert_weights_2[indices]

        ffn_biases_1 = self.expert_biases_1[indices]
        ffn_biases_2 = self.expert_biases_2[indices]

        x = x.unsqueeze(-1).unsqueeze(2)

        x = F.gelu((ffn_weights_1 @ x) + ffn_biases_1)
        x = F.gelu((ffn_weights_2 @ x) + ffn_biases_2)

        x = x.view(b, s, self.k, d)
        x = x * values.unsqueeze(-1)
        return torch.sum(x, dim=-2)


class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.moe = FFNNormal(dim)  # MoE(dim, num_experts=4) #FFNNormal(dim)
        self.attn = Attention(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, dim, num_layers=10):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, dim)
        self.positional_embedding = nn.Embedding(seq_len, dim)
        # self.rope = RoPE()
        self.layers = nn.Sequential(*[TransformerBlock(dim) for _ in range(num_layers)])
        self.to_logits = nn.Linear(dim, vocab_size)
        self.embedding_table.weight = self.to_logits.weight
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, target=None):
        B, T = x.shape
        self.pos_range = torch.arange(seq_len, device=x.device)
        x = self.embedding_table(x)
        # to get a batch dim
        x += self.positional_embedding(self.pos_range).unsqueeze(0)
        x = self.norm(self.layers(x))
        x = self.to_logits(x)  # shape is (batch_size, seq_len, vocab_size)

        x = x.view(B * T, vocab_size)
        y = target.view(B * T)

        loss = F.cross_entropy(x, y)
        return loss


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"Process {rank} of {world_size} is initialized.")
        dist.barrier()
    else:
        print("Distributed process group not initialized!")
        return

    print(f"Running model on {rank}.")
    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{device_id}")

    dataset = MathDataset()
    print("Successfully initialised the dataset")

    import random

    def get_batch():
        # Randomly choose batch_size indices without replacement.
        indices = random.sample(range(len(dataset)), batch_size)
        # Fetch the corresponding samples from the dataset.
        batch = [dataset[i] for i in indices]
        # Each sample is a tuple (input, target); unpack and stack them.
        data, target = zip(*batch)
        data = torch.stack(data).to(device).to(torch.long)
        target = torch.stack(target).to(device).to(torch.long)
        return data, target

    model = Transformer(dim).to(device)
    print("Initialising model...")
    ddp_model = DDP(
        model,
        device_ids=[device_id],
    )
    print("DDP succeeded")
    ddp_model.train()

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=3e-4)
    loss_avg = 0
    num_steps = 10000

    counter = 0

    for epoch in tqdm(range(num_steps), disable=(rank != 0), file=sys.stdout):
        try:
            for i in tqdm(range(1000), disable=(rank != 0), file=sys.stdout):
                if rank == 0 and i % 100 == 0:
                    print("Average loss:", loss_avg)
                    save_checkpoint(
                        ddp_model,
                        optimizer,
                        epoch,
                        i,
                        loss_avg,
                        f"checkpoint_{epoch}_{i}.pt",
                    )

                data, target = get_batch()
                loss: torch.Tensor = ddp_model(data, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_avg = (counter * loss_avg + loss.item()) / (counter + 1)
                counter += 1

        except Exception as e:
            print(f"An error occurred: {e}")

    dist.destroy_process_group()


if __name__ == "__main__":
    print("Starting main process...")
    main()
    print("Done!")
