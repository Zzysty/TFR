from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_size=1, encoded_space=4):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3),  # 编码器第一层
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=64, out_channels=encoded_space, kernel_size=3),  # 编码器第二层
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(encoded_space, out_channels=64, kernel_size=3),
            # 解码器第一层
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=3),
            # 解码器第二层
            nn.Sigmoid()  # 输出层使用sigmoid函数确保输出值在[0,1]之间
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
