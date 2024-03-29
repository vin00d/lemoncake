import math
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

class PositionalEncoding1(nn.Module):
    """
    Sinusoidal Positional Encoding
    """
    def __init__(self, d_model, dropout: float = 0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()                              # pe: (max_len, d_model)
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)                                  # (max_len, 1)  eg. (512, 1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()   # (d_model//2,) eg. (384,)
        a = position * div_term                                                                   # (max_len, d_model//2)  eg. (512, 384)
        pe[:, 0::2] = torch.sin(a)                            # seq = L[start:stop:step]    0::2 start 0, step_size = 2  
        pe[:, 1::2] = torch.cos(a)                            

        pe = pe.unsqueeze(0)                                                    # pe: (1, max_len, d_model)
        self.register_buffer('pe', pe)                                          # Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them.
                                                                                # register_buffer(name, tensor)
    def forward(self, x):                         # x: (B, seq_len)
        x = x + self.pe[:, :x.size(1)]             # (1, seq_len, d_model)
        return self.dropout(x)                     # (1, seq_len, d_model)


# Create a BERT model class in PyTorch with default values of 12 encoder layers, 12 attention heads, and hidden size 786
class BERT(nn.Module):
    def __init__(self, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        super(BERT, self).__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.dropout = dropout

        self.pos_encoder = PositionalEncoding1(d_model=hidden, dropout=dropout)

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
        x = self.pos_encoder(x)    
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

        # metrics
        # metrics = MetricCollection(
        #     [
        #         # Accuracy(num_classes=13),
        #         AUROC(task='multilabel', num_labels=13),
        #         # Precision(num_classes=13),
        #         # Recall(num_classes=13),
        #         # AveragePrecision(num_classes=13),
        #     ]
        # )
        # self.train_metrics = metrics.clone(prefix="train/")
        # self.valid_metrics = metrics.clone(prefix="valid/")
        # self.test_metrics = metrics.clone(prefix="test/")


    ## init weights from Karpathy's nanoGPT
    ## https://github.com/karpathy/nanoGPT/blob/a82b33b525ca9855d705656387698e13eb8e8d4b/model.py#L147
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # normal
            # nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.xavier_normal_(module.weight, gain=0.000001)
            # nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            # uniform
            # nn.init.uniform_(module.bias, a=-0.02, b=0.02)
            # nn.init.xavier_uniform_(module.weight, gain=0.0002)
            # nn.init.kaiming_uniform_(module.weight)
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

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.train_metrics.update(y_hat, y.int())
        # self.log_dict(self.train_metrics.compute(), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = self.valid_loss_fn(y_hat, y)

        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.valid_metrics.update(y_hat, y.int())
        # self.log_dict(self.valid_metrics.compute(), on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        # self.log("test_loss", loss, prog_bar=True)
        return 

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-5)

    def predict(self, x):
        y_hat = self(x)
        return y_hat
