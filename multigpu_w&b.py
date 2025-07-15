import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

import wandb

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8085"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        rank: int
    ) -> None:
        self.rank = rank
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(self.model, device_ids=[gpu_id])

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        epoch_loss = 0.0
        for step, (source, targets) in enumerate(self.train_data):
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss = self._run_batch(source, targets)
            epoch_loss += loss

            # Log per‐batch
            if self.rank == 0 and step % 10 == 0:
                wandb.log({
                    "batch_loss": loss,
                    "step": epoch * len(self.train_data) + step
                })

        avg_loss = epoch_loss / len(self.train_data)
        print(f"[GPU{self.gpu_id}] Epoch {epoch} – avg loss: {avg_loss:.4f}")

        # Log per‐epoch
        if self.rank == 0:
            wandb.log({"epoch": epoch, "avg_loss": avg_loss})

    def _save_checkpoint(self, epoch):
        ckp_path = f"checkpoint_epoch{epoch}.pt"
        torch.save(self.model.module.state_dict(), ckp_path)
        print(f"Epoch {epoch} | checkpoint saved at {ckp_path}")

        # Version with W&B Artifacts
        if self.rank == 0:
            artifact = wandb.Artifact(
                name=f"model-epoch-{epoch}",
                type="model",
                description=f"Checkpoint at epoch {epoch}"
            )
            artifact.add_file(ckp_path)
            wandb.log_artifact(artifact)

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.rank == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, gpu_id=rank, save_every=save_every, rank=rank)

    # Only rank 0 initializes W&B
    if rank == 0:
        wandb.init(
            project="my-ddp-project",
            config={
                "total_epochs": total_epochs,
                "save_every": save_every,
                "batch_size": batch_size,
                "lr": 1e-3,
                "world_size": world_size
            }
        )
        wandb.run.name = f"run-ddp-{wandb.run.id}"
        wandb.watch(trainer.model.module, log="all", log_freq=10)

    trainer.train(total_epochs)

    # Finish W&B on rank 0
    if rank == 0:
        wandb.finish()

    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)