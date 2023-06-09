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

# A function to initialize the weights of the model
def init_multimodalbert(m, initrange, zero_bn=False):
    """Initialize Multimodal BERT."""
    if isinstance(m, (nn.Embedding, nn.EmbeddingBag)):
        m.weight.data.uniform_(-initrange, initrange)
    if isinstance(m, (nn.TransformerEncoderLayer, nn.Linear)):
        for name, param in m.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.kaiming_normal_(param)
    if isinstance(m, (nn.BatchNorm1d)):
        nn.init.constant_(m.weight, 0.0 if zero_bn else 1.0)
    for l in m.children():
        init_multimodalbert(l, initrange, zero_bn)


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
        return self.linear(x[:, 0])  # x[:, 0]: (B, hidden) -> (B, 13)


class MultimodalBERT(pl.LightningModule):
    """
    Multimodal BERT
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(
        self,
        train_pos_wts,
        valid_pos_wts,
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
        self.save_hyperparameters()

        ## loss_fns
        self.train_loss_fn = nn.BCEWithLogitsLoss(pos_weight=train_pos_wts)
        self.valid_loss_fn = nn.BCEWithLogitsLoss(pos_weight=valid_pos_wts)

        ## model
        self.preprocessor = VectorPreProcessor(
            vector_width=vector_width, hidden=hidden, seq_len=seq_len
        )
        self.bert = BERT(
            hidden=hidden, n_layers=n_layers, attn_heads=attn_heads, dropout=dropout
        )
        self.predictor = MultiLabelPredictor(hidden=hidden)

        # init weights from Karpathy's nanoGPT (see below)
        self.apply(self._init_weights)

        ## metrics
        # metrics = MetricCollection(
        #     [
        #         Accuracy(num_classes=13),
        #         AUROC(num_classes=13),
        #         Precision(num_classes=13),
        #         Recall(num_classes=13),
        #         AveragePrecision(num_classes=13),
        #     ]
        # )
        # self.train_metrics = metrics.clone(prefix="train/")
        # self.valid_metrics = metrics.clone(prefix="valid/")
        # self.test_metrics = metrics.clone(prefix="test/")


    ## init weights from Karpathy's nanoGPT
    ## https://github.com/karpathy/nanoGPT/blob/a82b33b525ca9855d705656387698e13eb8e8d4b/model.py#L147
    def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                # torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            # elif isinstance(module, nn.Embedding):
            #     torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, x):  # x: (B, 4041)
        x = self.preprocessor(x)  # x: (B, 256, 768)
        x = self.bert(x)  # x: (B, seq_len, hidden)
        x = self.predictor(x)  # x: (B, 13)
        return x  # x: (B, 13)

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = self.train_loss_fn(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = self.valid_loss_fn(y_hat, y)

        self.log("valid_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        # self.log("test_loss", loss, prog_bar=True)
        return 

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def predict(self, x):
        y_hat = self(x)
        return y_hat
