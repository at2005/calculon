import torch
from torch import nn
import torch.distributed
import torch.nn.functional as F
import os

import torch.optim.lr_scheduler
from data import MathDataset
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from common import seq_len, vocab_size
import sys
import random

dropout = 0.25
dim = 1024
global_batch_size = 2**20
minibatch_size = 64
BATCH_ITERS = global_batch_size // minibatch_size
num_layers = 24

USE_CHECKPOINT = False


def load_checkpoint(model, checkpoint_file, device):
    checkpoint = torch.load(checkpoint_file, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    loss = checkpoint["loss"]
    step = checkpoint["step"]
    epoch = checkpoint["epoch"]
    optimizer_state = checkpoint["optimizer_state_dict"]
    scheduler_state = checkpoint["scheduler_state_dict"]

    return model, optimizer_state, scheduler_state, loss, epoch, step


def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, filename):
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, filename)


class RoPE(nn.Module):
    def __init__(self):
        super().__init__()
        indices = torch.arange(dim // 2)
        indices = -2 * (indices - 1) / dim
        thetas = 10_000**indices
        m = torch.arange(seq_len).view(-1, 1)
        vals = thetas * m  # (seq_len, d // 2)
        self.register_buffer("cosines", torch.cos(vals))
        self.register_buffer("sines", torch.sin(vals))

    def forward(self, x):
        output = torch.zeros_like(x)
        output[..., ::2] = self.cosines * x[..., ::2] - self.sines * x[..., 1::2]
        output[..., 1::2] = self.sines * x[..., ::2] + self.cosines * x[..., 1::2]
        return output


class RoPESingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = RoPE(dim, seq_len)
        return cls._instance


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=16):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.to_qkv = nn.Linear(dim, dim_head * 3 * heads, bias=False)
        self.to_out = nn.Linear(dim_head * heads, dim)
        self.keys_cache = None
        self.values_cache = None

    def reset_cache(self):
        self.keys_cache = None
        self.values_cache = None

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(
            3, dim=-1
        )  # shape is now (batch_size, seq_len, dim_head * heads) for q,k,v

        q = q.view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)

        rope = RoPESingleton.get_instance()
        q = rope(q)
        k = rope(k)

        if self.training:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=dropout
            )
        else:
            # cat along the seq dimension
            self.keys_cache = (
                torch.cat([self.keys_cache, k], -2)
                if self.keys_cache is not None
                else k
            )
            self.values_cache = (
                torch.cat([self.values_cache, v], -2)
                if self.values_cache is not None
                else v
            )
            out = F.scaled_dot_product_attention(
                q, self.keys_cache, self.values_cache, dropout_p=0.0, is_causal=False
            )

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.to_out(out)


class FFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim * 4, dim * 4),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.moe = FFN(dim)  # MoE(dim, num_experts=4) #FFN(dim)
        self.attn = Attention(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, dim)
        self.positional_embedding = nn.Embedding(seq_len, dim)
        # self.rope = RoPE()
        self.layers = nn.Sequential(*[TransformerBlock(dim) for _ in range(num_layers)])
        self.to_logits = nn.Linear(dim, vocab_size)
        self.embedding_table.weight = self.to_logits.weight
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, target=None, temperature=1.0):
        B, T = x.shape
        self.pos_range = torch.arange(T, device=x.device)
        x = self.embedding_table(x)
        # to get a batch dim
        x += self.positional_embedding(self.pos_range).unsqueeze(0)
        x = self.norm(self.layers(x))
        x = self.to_logits(x)  # shape is (batch_size, seq_len, vocab_size)

        if target is not None:
            x = x.view(B * T, vocab_size)
            y = target.view(B * T)

            loss = F.cross_entropy(x, y)
            return loss

        x = F.softmax(x[:, -1, :] / temperature, dim=-1)
        sample = torch.multinomial(x, num_samples=1)
        return sample


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

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{device_id}")

    dataset = MathDataset()

    def get_batch():
        # Randomly choose batch_size indices without replacement.
        indices = random.sample(range(len(dataset)), minibatch_size)
        # Fetch the corresponding samples from the dataset.
        batch = [dataset[i] for i in indices]
        # Each sample is a tuple (input, target); unpack and stack them.
        data, target = zip(*batch)
        data = torch.stack(data).to(device).to(torch.long)
        target = torch.stack(target).to(device).to(torch.long)
        return data, target

    loss_avg = 0
    num_steps = 10000
    counter = 0
    start_epoch = 0
    start_step = 0
    optimizer_state = None
    scheduler_state = None
    checkp_file = "checkpoint_7_200.pt"
    model = Transformer(dim, num_layers)

    if USE_CHECKPOINT and os.path.exists(checkp_file):
        model, optimizer_state, scheduler_state, loss_avg, start_epoch, start_step = (
            load_checkpoint(model, checkp_file, device)
        )
        counter = start_epoch * 1000 + start_step
        start_step += 1

    model = model.to(device)

    ddp_model = DDP(
        model,
        device_ids=[device_id],
    )

    compiled_ddp_model: nn.Module = torch.compile(
        model=ddp_model, mode="max-autotune", fullgraph=True
    )

    print("Model initialised. Starting to train")
    compiled_ddp_model.train()

    optimizer = torch.optim.AdamW(
        compiled_ddp_model.parameters(), lr=0.01, betas=(0.9, 0.95), weight_decay=0.1
    )

    linear_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=3e-2, total_iters=1000
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(1000 * (num_steps - 1)), eta_min=0.001
    )

    sequential_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[linear_scheduler, cosine_scheduler], milestones=[1000]
    )

    if USE_CHECKPOINT:
        optimizer.load_state_dict(optimizer_state)
        sequential_scheduler.load_state_dict(scheduler_state)

    for epoch in tqdm(
        range(start_epoch, num_steps), disable=(rank != 0), file=sys.stdout
    ):
        try:
            for i in tqdm(
                range(start_step, 1000), disable=(rank != 0), file=sys.stdout
            ):
                if rank == 0 and i % 100 == 0:
                    print("Average loss:", loss_avg)
                    save_checkpoint(
                        compiled_ddp_model,
                        optimizer,
                        sequential_scheduler,
                        epoch,
                        i,
                        loss_avg,
                        f"checkpoint_{epoch}_{i}.pt",
                    )

                optimizer.zero_grad()

                for _ in range(BATCH_ITERS):
                    data, target = get_batch()
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        loss: torch.Tensor = compiled_ddp_model(data, target)
                    loss = loss / BATCH_ITERS
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        compiled_ddp_model.parameters(), max_norm=1.0
                    )

                optimizer.step()
                sequential_scheduler.step()
                loss_avg = (counter * loss_avg + loss.item()) / (counter + 1)
                counter += 1

        except Exception as e:
            print(f"An error occurred: {e}")

    dist.destroy_process_group()


if __name__ == "__main__":
    print("Starting main process...")
    main()
    print("Done!")
