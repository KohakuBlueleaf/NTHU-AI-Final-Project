import torch
import torch.nn as nn

from hakuphi.trainer import BaseTrainer


def make_clip_grid(idx):
    """make idx grid and fill 1 if they have same idx"""
    grid = torch.meshgrid(idx, idx, indexing="xy")
    return (grid[0] == grid[1]).float() * 2 - 1


def l2_normalize(emb):
    return emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)


def log_sigmoid(x):
    return torch.log(torch.sigmoid(x))


class ContrastiveScorer(BaseTrainer):
    def __init__(self, embed_dim=768, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.embed_dim = embed_dim
        self.preference_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.Mish(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.summary_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.Mish(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.t = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.train_params = self.parameters()

    def forward(self, preference_emb, summary_emb):
        x_emb = l2_normalize(preference_emb)
        y_emb = l2_normalize(summary_emb)
        # Max 1, Min -1
        logits = x_emb @ y_emb.T * self.t + self.b
        return logits

    def predict(self, preference_emb, summary_emb):
        if preference_emb.dim < 2:
            preference_emb = preference_emb[None]
        logits = self.forward(preference_emb, summary_emb)
        return torch.argsort(logits, dim=-1, descending=True)

    def training_step(self, batch, batch_idx):
        preference, summary, idx = batch
        target = make_clip_grid(idx)

        preference_emb = self.preference_proj(preference)
        summary_emb = self.summary_proj(summary)
        x_emb = l2_normalize(preference_emb)
        y_emb = l2_normalize(summary_emb)
        # Max 1, Min -1
        logits = x_emb @ y_emb.T * self.t + self.b

        # Want to get all 1, (logit1 with target1, or logit-1 with target-1)
        loss = -torch.mean(log_sigmoid(logits * target))

        if self._trainer is not None:
            self.log("train/loss", loss, on_step=True, logger=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_loss = 0
        self.val_batch = 0

    @torch.no_grad()
    def validation_step(self, batch, idx):
        self.eval()
        preference, summary, idx = batch
        target = make_clip_grid(idx)

        preference_emb = self.preference_proj(preference)
        summary_emb = self.summary_proj(summary)
        x_emb = l2_normalize(preference_emb)
        y_emb = l2_normalize(summary_emb)
        # Max 1, Min -1
        logits = x_emb @ y_emb.T * self.t + self.b

        # Want to get all 1, (logit1 with target1, or logit-1 with target-1)
        loss = -torch.mean(log_sigmoid(logits * target))
        self.val_loss += loss.item()
        self.val_batch += 1

    def on_validation_epoch_end(self) -> None:
        val_loss = self.val_loss / self.val_batch
        if self._trainer is not None:
            self.log("val/loss", val_loss, logger=True, prog_bar=True)
