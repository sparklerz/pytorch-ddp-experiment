import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import argparse

import mlflow
import mlflow.pytorch


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
        is_master: bool
    ) -> None:
        self.gpu_id = gpu_id
        self.is_master = is_master
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model.to(gpu_id), device_ids=[gpu_id])
        self.global_step = 0
        # Cache batch-size once
        self.b_sz = len(next(iter(self.train_data))[0])

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

        if self.is_master:                                # log only once
            mlflow.log_metric("train_loss", loss.item(), step=self.global_step)
        self.global_step += 1

    def _run_epoch(self, epoch):
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {self.b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id, non_blocking=True)
            targets = targets.to(self.gpu_id, non_blocking=True)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        PATH = f"checkpoint_epoch_{epoch}.pt"
        torch.save(self.model.module.state_dict(), PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

        if self.is_master:
            mlflow.log_artifact(PATH)
            mlflow.pytorch.log_model(
                self.model.module,
                artifact_path="model",
                registered_model_name="demo_ddp_model"
            )

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
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


def _default_tracking_uri(port="8082"):
    return f"http://127.0.0.1:{port}"


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    is_master = rank == 0

    if is_master:
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", _default_tracking_uri())
        )
        mlflow.set_experiment("pytorch_ddp_demo")
        mlflow.start_run()
        mlflow.log_params({
            "epochs": total_epochs,
            "batch_size": batch_size,
            "optimizer": "SGD",
            "lr": 1e-3,
            "world_size": world_size
        })

    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every, is_master)
    trainer.train(total_epochs)

    if is_master:
        mlflow.end_run()

    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed PyTorch + MLflow")
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size, join=True)