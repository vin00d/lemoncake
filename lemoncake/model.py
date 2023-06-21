import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torchmetrics import (
    MetricCollection,
    Accuracy,
    AUROC,
    Precision,
    Recall,
    AveragePrecision,
)


# Create a BERT model class in PyTorch with default values of 12 encoder layers, 12 attention heads, and hidden size 786
class BERT(nn.Module):
    def __init__(self, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        super(BERT, self).__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.dropout = dropout

        # BERT consists of a stack of 12 identical encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=attn_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        # x: [batch_size, seq_len, hidden]

        # print(f"x.shape from BERT: {x.shape}")
        x = self.encoder(x)
        return x


class VectorPreProcessor(nn.Module):
    """
    Preprocesses pretrained vectors before sending to BERT
    """

    def __init__(self, vector_width=4041, hidden=768, seq_len=256):
        """
        :param vector_width: pretrained vector width
        :param hidden: BERT model hidden size
        :param seq_len: BERT model sequence length
        """
        super().__init__()
        self.hidden = hidden
        self.seq_len = seq_len
        self.linear = nn.Linear(
            vector_width, hidden * seq_len
        )  # 4041 -> 768*256 = 196,608

    def forward(self, x):
        # print(f"x.shape from VectorPreProcessor: {x.shape}")
        x = self.linear(x)  # 4041 -> 768*256 = 196,608
        return x.view(x.shape[0], self.seq_len, self.hidden)  # 196,608 -> (B, 256, 768)


class MultiLabelPredictor(nn.Module):
    """
    13-class multi label classification model on top of BERT
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 13)
        # self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # print(f"x.shape from MultiLabelPredictor: {x.shape}")
        return self.linear(x[:, 0])  # x[:, 0]: (B, hidden) -> (B, 14)


class MultimodalBERT(pl.LightningModule):
    """
    Multimodal BERT
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(
        self,
        vector_width=4041,
        seq_len=256,
        hidden=768,
        n_layers=12,
        attn_heads=12,
        dropout=0.1,
    ):
        """
        :param bert: BERT model which should be trained
        """

        super().__init__()
        self.preprocessor = VectorPreProcessor(
            vector_width=vector_width, hidden=hidden, seq_len=seq_len
        )
        self.bert = BERT(
            hidden=hidden, n_layers=n_layers, attn_heads=attn_heads, dropout=dropout
        )
        self.predictor = MultiLabelPredictor(hidden=hidden)

    def forward(self, x):  # x: (B, 4041)
        x = self.preprocessor(x)  # x: (B, 256, 768)
        x = self.bert(x)  # x: (B, seq_len, hidden)
        x = self.predictor(x)  # x: (B, 14)
        return x  # x: (B, 14)

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def predict(self, x):
        y_hat = self(x)
        return y_hat
