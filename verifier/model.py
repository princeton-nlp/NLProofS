import numpy as np
import torchmetrics
from common import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from transformers import AutoTokenizer, AutoModel


class EntailmentClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        lr: float,
        warmup_steps: int,
        pos_weight: float,
        max_input_len: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.pos_weight = pos_weight
        self.max_input_len = max_input_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_input_len)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.encoder.config.hidden_size, 1)
        self.metrics = {
            "train": {
                "accuracy": torchmetrics.Accuracy(threshold=0),
                "average_precision": torchmetrics.AveragePrecision(pos_label=1),
                "precision": torchmetrics.Precision(),
                "recall": torchmetrics.Recall(),
                "specificity": torchmetrics.Specificity(),
                "f1": torchmetrics.F1Score(),
            },
            "val": {
                "accuracy": torchmetrics.Accuracy(threshold=0),
                "average_precision": torchmetrics.AveragePrecision(pos_label=1),
                "precision": torchmetrics.Precision(),
                "recall": torchmetrics.Recall(),
                "specificity": torchmetrics.Specificity(),
                "f1": torchmetrics.F1Score(),
            },
        }
        for split, metrics in self.metrics.items():
            for name, m in metrics.items():
                self.add_module(f"{name}_{split}", m)

    def log_metrics(self, split: str, logit: torch.Tensor, label: torch.Tensor) -> None:
        for (name, metric) in self.metrics[split].items():
            metric(logit, label)
            self.log(f"{name}_{split}", metric, on_step=False, on_epoch=True)

    def forward(  # type: ignore
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Any:
        features = self.encoder(input_ids, attention_mask).pooler_output
        return self.fc(features).squeeze(dim=1)

    def on_train_start(self) -> None:
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)  # type: ignore
            assert self.trainer is not None
            print(f"Logging to {self.trainer.log_dir}")

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:  # type: ignore
        logit = self(batch["input_ids"], batch["attention_mask"])
        loss = F.binary_cross_entropy_with_logits(
            logit, batch["label"].float(), pos_weight=torch.tensor(self.pos_weight)
        )
        self.log("loss_train", loss, on_epoch=True)
        self.log_metrics("train", logit, batch["label"])
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:  # type: ignore
        logit = self(batch["input_ids"], batch["attention_mask"])
        loss = F.binary_cross_entropy_with_logits(
            logit, batch["label"].float(), pos_weight=torch.tensor(self.pos_weight)
        )
        self.log("loss_val", loss)
        self.log_metrics("val", logit, batch["label"])

    def configure_optimizers(self) -> Dict[str, Any]:
        assert self.trainer is not None
        max_steps = (
            self.trainer.max_epochs
            * len(self.trainer.datamodule.train_dataloader())  # type: ignore
            // self.trainer.accumulate_grad_batches
        )
        return get_optimizers(
            self.parameters(),
            self.lr,
            self.warmup_steps,
            max_steps,
        )

    def score(self, premises: List[str], conclusion: str) -> float:
        entailment = self.tokenizer(
            ". ".join(premises) + ".",
            conclusion,
            padding="longest",
            truncation="longest_first",
            max_length=self.max_input_len,
            return_tensors="pt",
        )
        input_ids = entailment.input_ids.to(self.device)
        attention_mask = entailment.attention_mask.to(self.device)
        logit = torch.sigmoid(self(input_ids, attention_mask))
        return logit.detach().item()

    def batch_score(
        self, premises_batch: List[List[str]], conclusion_batch: List[str]
    ) -> Any:
        assert len(premises_batch) == len(conclusion_batch)
        if len(premises_batch) == 0:
            return np.array([])
        entailment = self.tokenizer(
            [". ".join(premises) + "." for premises in premises_batch],
            conclusion_batch,
            padding="longest",
            truncation="longest_first",
            max_length=self.max_input_len,
            return_tensors="pt",
        )
        input_ids = entailment.input_ids.to(self.device)
        attention_mask = entailment.attention_mask.to(self.device)
        logits = torch.sigmoid(self(input_ids, attention_mask))
        return logits.detach().cpu().numpy()  # type: ignore
