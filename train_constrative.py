import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

from dataset import final

import torch

torch.set_float32_matmul_precision("medium")
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sch
import torch.utils.data as Data

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from transformers import AutoModel

from modules.contrastive import ContrastiveScorer


EPOCH = 10
GPUS = 1
BATCH_SIZE = 512
GRAD_ACC = 1


def load_trainer(lr=1e-3, warmup_steps=100, t_max=1000_000) -> ContrastiveScorer:
    if isinstance(warmup_steps, float):
        assert 0.0 <= warmup_steps <= 1.0
        warmup_steps = int(warmup_steps * t_max)
    return ContrastiveScorer(
        768,
        name="Jina-Emb-Contrastive",
        lr=lr,
        optimizer=optim.AdamW,
        opt_configs={
            "weight_decay": 0.05,
            "betas": (0.9, 0.999),
        },
        lr_scheduler=lr_sch.CosineAnnealingLR,
        lr_sch_configs={
            "T_max": t_max - warmup_steps,
            "eta_min": lr * 1e-2,
        },
        use_warm_up=bool(warmup_steps),
        warm_up_period=warmup_steps,
    )


def load_final_dataset(emb_model, split="train-en"):
    raw_datas = final.load(split)["train"]
    processor = final.contrastive_processor(emb_model)
    dataset = raw_datas.shuffle().map(processor, desc="load data", batch_size=256)
    return dataset


def main():
    # Loading models and datasets
    emb_model = (
        AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True
        )
        .half()
        .cuda()
    )

    # Setup dataset
    main_dataset = load_final_dataset(emb_model)
    val_dataset = load_final_dataset(emb_model, "test-en")
    dataset = Data.ConcatDataset([main_dataset])

    trainer_module = load_trainer(
        1e-2,
        0.1,
        len(dataset) * EPOCH // (BATCH_SIZE * GPUS * GRAD_ACC),
    )
    print(f"Total training step: {len(dataset)*EPOCH//(BATCH_SIZE*GPUS*GRAD_ACC)}")

    def collate(batch):
        return [
            torch.tensor([x["preference"] for x in batch]),
            torch.tensor([x["summary"] for x in batch]),
            torch.tensor([x["idx"] for x in batch]),
        ]

    data_loader = Data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=collate,
    )
    val_loader = Data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=BATCH_SIZE,
        collate_fn=collate,
    )

    # Train!
    logger = None
    logger = WandbLogger(
        name="jina-siglip-test",
        project="Contrastive",
        # offline = True,
    )
    trainer = pl.Trainer(
        precision="16-mixed",
        accelerator="gpu",
        devices=GPUS,
        max_epochs=EPOCH,
        logger=logger,
        log_every_n_steps=1,
        accumulate_grad_batches=GRAD_ACC,
        callbacks=[
            # ProdigyLRMonitor(logging_interval="step"),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(every_n_epochs=1),
        ],
        gradient_clip_val=1.0,
        # fast_dev_run=True
    )
    trainer.fit(
        trainer_module.train(),
        train_dataloaders=data_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    pl.seed_everything(3407)
    main()
