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
from common import dim, minibatch_size, BATCH_ITERS, num_layers, USE_CHECKPOINT
import sys
import random
from models import Transformer


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


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"Process {rank} of {world_size} is initialized.")
    else:
        print("Distributed process group not initialized!")
        return

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
    model.train()

    compiled_model = torch.compile(model=model, mode="max-autotune", fullgraph=True)

    print("Finished compiling model")

    compiled_ddp_model = DDP(
        compiled_model,
        device_ids=[device_id],
    )

    print("Model initialised. Starting to train")

    optimizer = torch.optim.AdamW(
        compiled_ddp_model.parameters(),
        lr=torch.tensor(0.01),
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    linear_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=torch.tensor(3e-2), total_iters=1000
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(1000 * (num_steps - 1)), eta_min=torch.tensor(0.001)
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

                for _ in tqdm(range(BATCH_ITERS)):
                    data, target = get_batch()
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
