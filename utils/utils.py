import torch
import math
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self):
        super().__init__(Attention)
    def forward(self,q, k, v, d_k, attn_mask=None):
        scores = torch.bmm(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        attn = nn.Softmax(scores)
        outputs = torch.bmm(attn, v)
        return outputs, attn

class PointerNetwok(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attention = Attention()
        self.Encoder = nn.LSTM(input_size=256, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)
        self.Decoder = nn.LSTM(input_size=256, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)
        self.embedding = nn.Embedding(256, 256)
    def forward(self, inputs):
        






