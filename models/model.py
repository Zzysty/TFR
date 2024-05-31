import random

import torch
import torch.nn as nn

from models.autoencoder import Autoencoder
from models.tcn import TemporalConvNet


# -----------------------------------------LSTM-----------------------------------------
class LSTM(nn.Module):
    def __init__(self, num_features, hidden_units, num_layers, output_size, dropout_rate):
        super(LSTM, self).__init__()
        self.hidden_units = hidden_units
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        # Fully connected layer
        self.linear = nn.Linear(in_features=hidden_units, out_features=output_size)

    def forward(self, x):
        # Forward propagate LSTM and get final state
        lstm_out, _ = self.lstm(x)  # 默认会在x的设备上创建隐藏状态

        # Only take the output from the final timestep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[:, -1, :])
        return y_pred


# -----------------------------------------GRU-----------------------------------------
class GRU(nn.Module):
    def __init__(self, num_features, hidden_units, num_layers, output_size, dropout_rate):
        super(GRU, self).__init__()

        self.rnn = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.linear = nn.Linear(hidden_units, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        y_pred = self.linear(out[:, -1, :])
        return y_pred


# -----------------------------------------TCN-----------------------------------------
class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, output_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # 调整输入数据的维度，使其符合卷积层的期望
        # 从 [batch_size, sequence_length, num_features] 到 [batch_size, num_features, sequence_length]
        x = x.permute(0, 2, 1)

        tcn_out = self.tcn(x)
        tcn_out = tcn_out[:, :, -1]
        out = self.linear(tcn_out)

        return out


# --------------------------------------AE-TCN-----------------------------------------
class TCN_Autoencoder(nn.Module):
    def __init__(self, input_size, encoded_space, num_channels, kernel_size, output_size, dropout):
        super(TCN_Autoencoder, self).__init__()
        self.autoencoder = Autoencoder(input_size, encoded_space)
        self.tcn = TemporalConvNet(encoded_space, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # 从 [batch_size, sequence_length, num_features] 到 [batch_size, num_features, sequence_length]
        x = x.permute(0, 2, 1)  # [32, 30, 47] -> [32, 47, 30]
        _, encoded = self.autoencoder(x)
        tcn_out = self.tcn(encoded)
        tcn_out = tcn_out[:, :, -1]  # 取最后一个时间步的特征
        out = self.linear(tcn_out)
        return out


# -----------------------------------------MLP-----------------------------------------
class MLP(nn.Module):
    def __init__(self, input_size, hidden_units, output_size, dropout_rate):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units, output_size)
        )

    def forward(self, x):
        # 展平输入数据
        # 假设原始数据形状为 [batch_size, sequence_length, num_features]
        # 展平为 [batch_size, sequence_length * num_features]
        x = x.reshape(x.size(0), -1)

        # 通过 MLP
        out = self.mlp(x)
        return out


# -----------------------------------------seq2seq-----------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        output, (h, c) = self.lstm(input_seq, (h_0, c_0))

        return h, c


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, input_seq, h, c):
        # input_seq(batch_size, input_size)
        input_seq = input_seq.unsqueeze(1)
        output, (h, c) = self.lstm(input_seq, (h, c))
        # output(batch_size, seq_len, num * hidden_size)
        pred = self.linear(output.squeeze(1))  # pred(batch_size, 1, output_size)

        return pred, h, c


class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.Encoder = Encoder(input_size, hidden_size, num_layers, batch_size)
        self.Decoder = Decoder(input_size, hidden_size, num_layers, output_size, batch_size)

    def forward(self, input_seq):
        target_len = self.output_size  # 预测步长
        batch_size, seq_len, _ = input_seq.shape[0], input_seq.shape[1], input_seq.shape[2]
        h, c = self.Encoder(input_seq)
        outputs = torch.zeros(batch_size, self.input_size, self.output_size).to(device)
        decoder_input = input_seq[:, -1, :]
        for t in range(target_len):
            decoder_output, h, c = self.Decoder(decoder_input, h, c)
            outputs[:, :, t] = decoder_output
            decoder_input = decoder_output

        return outputs[:, 0, :]
