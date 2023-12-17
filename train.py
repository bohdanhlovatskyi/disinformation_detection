import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers import BertModel, AutoTokenizer
from torch.utils.data import DataLoader
from typing import Optional

from data import create_classification_dataset

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

class DisinformationBERT(pl.LightningModule):
    def __init__(
        self, 
        pretrain_weights: str = "ai-forever/ruBert-base", 
        num_classes: int = 3, 
        num_lin_blocks: int = 1, 
        dropout_prob: float = 0.3,
        learning_rate: float = 1e-5, 
    ):
        super(DisinformationBERT, self).__init__()

        self.learning_rate = learning_rate

        self.model = BertModel.from_pretrained(pretrain_weights, cache_dir=".")
        self.model.requires_grad_(False)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_weights, cache_dir=".")

        classification_head = []
        for i in range(num_lin_blocks):
            classification_head.append(torch.nn.Dropout(dropout_prob))
            if i != num_lin_blocks - 1:
                classification_head.append(
                    torch.nn.Linear(
                        self.model.config.hidden_size,
                        self.model.config.hidden_size)
                    )

                classification_head.append(torch.nn.GELU())
            else:
                classification_head.append(torch.nn.Linear(self.model.config.hidden_size, num_classes))

        self.classification_head = nn.Sequential(*classification_head)
    
        # self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.loss_fn = FocalLoss(alpha=1, gamma=2, reduction='mean')

        self.val_f1 = torchmetrics.F1Score(
            task="multiclass",
            num_classes=num_classes, 
            average="macro"
        )

    def forward(self, input_ids, attention_mask):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.classification_head(pooled_output)

        return output.squeeze(1)

    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, labels = batch.values()
        predictions = self(input_ids, attention_mask)
        loss = self.loss_fn(predictions, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, labels = batch.values()
        predictions = self(input_ids, attention_mask)

        loss = self.loss_fn(predictions, labels)
        self.val_f1(torch.argmax(predictions, dim=1), labels)
        
        self.log_dict({
            "val_loss": loss, 
            "val_f1": self.val_f1
        }, on_epoch=True, on_step=False, prog_bar=True)
        
        return {"predictions": predictions, "labels": labels, "val_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer), 
                "monitor": "val_loss",
            },
        }

if __name__ == "__main__":
    pl.seed_everything(42)
    model = DisinformationBERT()

    train, val = create_classification_dataset("train.csv", model.tokenizer)

    train_loader = DataLoader(train, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val, batch_size=4, drop_last=True, shuffle=False, num_workers=2)

    logger = pl.loggers.TensorBoardLogger(
        save_dir=os.getcwd(),
        version=3,
        name="lightning_logs"
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints3',
        filename='{epoch:02d}-{val_loss:.2f}', 
        save_top_k=3, 
    )

    trainer = pl.Trainer(
        # fast_dev_run=True, 
        devices=5, 
        accelerator="auto", 
        # log_every_n_steps=10, 
        logger=logger, 
        # precision="16-true", 
        # check_val_every_n_epoch=10, 
        callbacks=[
            checkpoint_callback, 
            pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        ]
    )

    trainer.fit(
        model,
        train_loader,
        val_loader, 
    )

    print(checkpoint_callback.best_k_models)
