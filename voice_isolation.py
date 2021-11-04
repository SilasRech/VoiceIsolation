import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl




class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        s = self.decoder(x)
        return s

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x = batch[0]
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=512, bottleneck=512, kernel_size=16):
        super(Encoder, self).__init__()
        self.encoder = nn.Conv1d(
            in_channels, out_channels, kernel_size, kernel_size//2, padding=0)
        self.conv1x1 = nn.Conv1d(out_channels, bottleneck, 1)

    def forward(self, x):
        if x.dim() >= 3:
            raise RuntimeError(
                "{} accept 1/2D tensor as input, but got {:d}".format(
                    self.__name__, x.dim()))
        # when inference, only one utt
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
        if x.dim() == 2:
            x = torch.unsqueeze(x, 1)
        x = self.encoder(x)
        w = self.conv1x1(x)
        return w


class Decoder(nn.Module):
    def __init__(self, in_channels=512, out_channels=1, kernel_size=16):
        super(Decoder, self).__init__()
        self.decoder = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.decoder(x)
        return torch.squeeze(x)
