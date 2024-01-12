import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_size = 128
        self.embedding_size = 128
        self.embedding = nn.Embedding(100, self.embedding_size)
        self.encoder = nn.LSTM(self.embedding_size, self.hidden_size, batch_first=True)
        self.decoder = nn.LSTM(self.embedding_size, self.hidden_size, batch_first=True)
        self.W1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.vt = nn.Linear(self.hidden_size, 1)
        self.W = nn.Linear(self.hidden_size * 2, self.embedding_size)
        self.V = nn.Linear(self.embedding_size, 1)
         
    def forward(self, inputs):
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        embedded = self.embedding(inputs)
        encoder_outputs, (h_n, c_n) = self.encoder(embedded)
        decoder_outputs, (h_n, c_n) = self.decoder(embedded, (h_n, c_n))
        encoder_outputs = encoder_outputs.transpose(1, 2)
        decoder_outputs = decoder_outputs.transpose(1, 2)
        # (batch_size, seq_len, hidden_size) * (batch_size, hidden_size, seq_len) -> (batch_size, seq_len, seq_len)
        G = torch.bmm(decoder_outputs, encoder_outputs)
        G = G.transpose(1, 2)
        # (batch_size, seq_len, seq_len)
        G = F.tanh(self.W1(G) + self.W2(decoder_outputs))
        # (batch_size, seq_len, seq_len)
        G = self.vt(G).squeeze(2)
        # (batch_size, seq_len)
        alpha = F.softmax(G, dim=1)
        # (batch_size, seq_len, hidden_size) * (batch_size, seq_len, 1) -> (batch_size, hidden_size)
        z = torch.bmm(encoder_outputs, alpha.unsqueeze(2)).squeeze(2)
        # (batch_size, hidden_size)
        h_star = F.tanh(self.W(torch.cat([z, h_n.squeeze(0)], dim=1)))
        # (batch_size, 1)
        out = self.V(h_star).squeeze(1)
        return out
    
    def predict(self, inputs):
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        embedded = self.embedding(inputs)
        encoder_outputs, (h_n, c_n) = self.encoder(embedded)
        decoder_input = embedded[:, 0, :].unsqueeze(1)
        probs = []
        for i in range(seq_len):
            decoder_output, (h_n, c_n) = self.decoder(decoder_input, (h_n, c_n))
            decoder_output = decoder_output.transpose(1, 2)
            encoder_outputs = encoder_outputs.transpose(1, 2)
            # (batch_size, 1, hidden_size) * (batch_size, hidden_size, seq_len) -> (batch_size, 1, seq_len)
            G = torch.bmm(decoder_output, encoder_outputs)
            G = G.transpose(1, 2)
            # (batch_size, 1, seq_len)
            G = F.tanh(self.W1(G) + self.W2(decoder_output))
            # (batch_size, 1, seq_len)
            G = self.vt(G).squeeze(2)
            # (batch_size, seq_len)
            alpha = F.softmax(G, dim=1)
            # (batch_size, 1, hidden_size) * (batch_size, seq_len, 1) -> (batch_size, 1, 1)
            z = torch.bmm(encoder_outputs, alpha.unsqueeze(2)).squeeze(2)
            # (batch_size, hidden_size)
            h_star = F.tanh(self.W(torch.cat([z, h_n.squeeze(0)], dim=1)))
            # (batch_size, 1)
            out = self.V(h_star).squeeze(1)
            probs.append(out)
            decoder_input = embedded[:, i, :].unsqueeze(1)
        probs = torch.stack(probs, dim=1)
        return probs