import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram

from dataclasses import dataclass
import gin
import math
import torch.nn.functional as F
from typing import Tuple
import numpy as np


class Conv1dSamePaddingReflect(nn.Module):
    """
    1D Convolution module with 'same' padding (reflect mode).

    Attributes:
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        dilation (int): Dilation rate of the convolution.
        conv (nn.Conv1d): Convolution module.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """
        Initialize a Conv1dSamePaddingReflect module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolution kernel.
            stride (int): Stride of the convolution. Defaults to 1.
            dilation (int): Dilation rate of the convolution. Defaults to 1.
            groups (int): Number of groups for grouped convolution. Defaults to 1.
            bias (bool): Whether to include a bias. Defaults to True.

        Returns:
            None
        """
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Determine padding
        L_in = x.size(-1)
        L_out = (math.floor((L_in - self.dilation *
                             (self.kernel_size - 1) - 1) / self.stride) + 1)
        padding = (L_in - L_out) // 2

        x = F.pad(x, (padding, padding), mode="reflect")

        return self.conv(x)


class TDNNBlock(nn.Module):
    """
    Time-Delay Neural Network (TDNN) module.

    Attributes:
        conv (Conv1dSamePaddingReflect): Convolution module.
        activation (nn.ReLU): Activation module.
        norm (nn.BatchNorm1d): Normalization module.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        groups: int = 1,
    ):
        """
        Initialize a TDNNBlock module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            dilation (int): Dilation rate for the convolution.
            groups (int): Groups for the convolution. Defaults to 1.

        Returns:
            None
        """
        super().__init__()

        self.conv = Conv1dSamePaddingReflect(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
        )
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.norm(self.activation(self.conv(x)))


class Res2NetBlock(nn.Module):
    """
    Res2Net module.

    Attributes:
        blocks (nn.ModuleList): List of TDNNBlock modules.
        scale (int): Scale factor for the number of channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int = 8,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        """
        Initialize a Res2NetBlock module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            scale (int): Scale factor for the number of channels. Defaults to 8.
            kernel_size (int): Size of the kernel for the TDNN blocks. Defaults to 3.
            dilation (int): Dilation factor for the TDNN blocks. Defaults to 1.

        Raises:
            AssertionError: If input or output channels are not divisible by the scale factor.
        """
        super().__init__()

        assert in_channels % scale == 0
        assert out_channels % scale == 0

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        self.blocks = nn.ModuleList([
            TDNNBlock(
                in_channel,
                hidden_channel,
                kernel_size=kernel_size,
                dilation=dilation,
            ) for i in range(scale - 1)
        ])
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        y = []

        chunks = torch.chunk(x, self.scale, dim=1)
        y_i = chunks[0]
        y.append(y_i)

        for i, block in enumerate(self.blocks):
            x_i = chunks[i + 1]
            if i == 0:
                y_i = block(x_i)
            else:
                y_i = block(x_i + y_i)
            y.append(y_i)
        y = torch.cat(y, dim=1)
        return y


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) module.

    Attributes:
        conv1 (Conv1dSamePaddingReflect): First convolution module.
        conv2 (Conv1dSamePaddingReflect): Second convolution module.
        relu (nn.ReLU): First activation module.
        sigmoid (nn.Sigmoid): Second activation module.
    """

    def __init__(self, in_channels: int, se_channels: int, out_channels: int):
        """
        Initialize a SEBlock module.

        Args:
            in_channels (int): Number of input channels.
            se_channels (int): Number of SE channels.
            out_channels (int): Number of output channels.

        Returns:
            None
        """
        super().__init__()

        self.conv1 = Conv1dSamePaddingReflect(in_channels=in_channels,
                                              out_channels=se_channels,
                                              kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = Conv1dSamePaddingReflect(in_channels=se_channels,
                                              out_channels=out_channels,
                                              kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        s = x.mean(dim=2, keepdim=True)
        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))
        return s * x


class SERes2NetBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Res2Net module.

    Attributes:
        tdnn1 (TDNNBlock): First TDNN module.
        res2net_block (Res2NetBlock): Res2NetBlock module.
        tdnn2 (TDNNBlock): Second TDNN module
        se_block (SEBlock): SEBlock module.
        shortcut (Optional[Conv1dSamePaddingReflect]): Residual connection module.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        res2net_scale: int = 8,
        se_channels: int = 128,
        kernel_size: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ):
        """
        Initialize a SERes2NetBlock module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            res2net_scale (int): Scale factor for the number of channels in Res2Net. Defaults to 8.
            se_channels (int): Number of channels for Squeeze-and-Excitation. Defaults to 128.
            kernel_size (int): Size of the kernel for Res2Net convolution. Defaults to 1.
            dilation (int): Dilation rate for the Res2Net module convolution. Defaults to 1.
            groups (int): Groups for the Res2Net module convolution. Defaults to 1.

        Returns:
            None
        """
        super().__init__()

        self.tdnn1 = TDNNBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            groups=groups,
        )
        self.res2net_block = Res2NetBlock(out_channels, out_channels,
                                          res2net_scale, kernel_size, dilation)
        self.tdnn2 = TDNNBlock(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            groups=groups,
        )
        self.se_block = SEBlock(out_channels, se_channels, out_channels)

        if in_channels != out_channels:
            self.shortcut = Conv1dSamePaddingReflect(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        residual = x
        # if self.shortcut:

        residual = self.shortcut(x)

        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x)

        return x + residual


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive Statistics Pooling (ASP) module.

    Attributes:
        global_context (bool): Whether to use global context.
        tdnn (TDNNBlock): TDNN module.
        tanh (nn.Tanh): Activation module.
        conv (Conv1dSamePaddingReflect): Convolution module.
    """

    def __init__(
        self,
        channels: int,
        attention_channels: int = 128,
        global_context: bool = True,
    ):
        """
        Initialize an AttentiveStatisticsPooling module.

        Args:
            channels (int): Number of input channels.
            attention_channels (int): Number of attention channels. Defaults to 128.
            global_context (bool): Whether to use global context. Defaults to True.

        Returns:
            None
        """
        super().__init__()

        self.global_context = global_context

        in_channels = channels * 3 if global_context else channels

        self.tdnn = TDNNBlock(in_channels, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = Conv1dSamePaddingReflect(in_channels=attention_channels,
                                             out_channels=channels,
                                             kernel_size=1)

    def _compute_statistics(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        eps: float = 1e-12,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the statistics of a tensor.

        Args:
            x (torch.Tensor): Input tensor. Shape: (N, L, D).
            m (torch.Tensor): Mask tensor. Shape: (N, L).
            eps (float): Small value to prevent division by zero. Defaults to 1e-12.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and standard deviation tensors. Shape: (N, L).
        """
        mean = (m * x).sum(dim=2)
        std = torch.sqrt(
            (m * (x - mean.unsqueeze(dim=2)).pow(2)).sum(dim=2).clamp(eps))
        return mean, std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.global_context:
            L = x.size(-1)
            mean, std = self._compute_statistics(x, torch.tensor(1 / L))
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x

        attn = self.conv(self.tanh(self.tdnn(attn)))
        attn = F.softmax(attn, dim=2)

        mean, std = self._compute_statistics(x, attn)

        stats = torch.cat((mean, std), dim=1)
        stats = stats.unsqueeze(dim=2)

        return stats


@gin.configurable
class ECAPATDNN(nn.Module):
    """
    Emphasized Channel Attention, Propagation and Aggregation in TDNN (ECAPA-TDNN) encoder.

    Paper:
        ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification
        *Brecht Desplanques, Jenthe Thienpondt, Kris Demuynck*
        INTERSPEECH 2020
        https://arxiv.org/abs/2005.07143

    Attributes:
        pooling (bool): Whether to apply temporal pooling.
        blocks (nn.ModuleList): List of TDNNBlock and SERes2NetBlock modules.
        mfa (TDNNBlock): Multi-Frame Aggregation (MFA) module.
        asp (AttentiveStatisticsPooling): Attentive Statistics Pooling (ASP) module.
        asp_bn (nn.BatchNorm1d): Batch normalization module for Attentive Statistics Pooling (ASP).
        fc (Conv1dSamePaddingReflect): Final fully-connected layer.
    """

    def __init__(self,
                 in_size: int,
                 channels,
                 kernel_size,
                 res2net_scale,
                 se_channels,
                 attention_channels,
                 global_context,
                 pooling,
                 use_tanh,
                 spherical_normalization,
                 ac_regularisation,
                 dilation=1,
                 groups=1):
        """
        Initialize an ECAPA-TDNN encoder.

        Args:
            ...
        Returns:
            None
        """
        super().__init__()
        self.out_dim = channels.pop(-1)
        last_channel = int(np.sum([c for c in channels[1:]]))
        self.channels = channels + [last_channel]

        self.kernel_sizes = [kernel_size] * len(self.channels)
        self.dilations = [dilation] * len(self.channels)
        self.groups = [groups] * len(self.channels)
        self.res2net_scale = res2net_scale
        self.se_channels = se_channels
        self.attention_channels = attention_channels
        self.global_context = global_context
        self.pooling = pooling

        self.use_tanh = use_tanh
        self.spherical_normalisation = spherical_normalization
        self.ac_regularisation = ac_regularisation

        assert len(self.channels) == len(self.kernel_sizes)
        assert len(self.channels) == len(self.dilations)

        self.blocks = nn.ModuleList()

        self.blocks.append(
            TDNNBlock(
                in_size,
                self.channels[0],
                self.kernel_sizes[0],
                self.dilations[0],
                self.groups[0],
            ))

        for i in range(1, len(self.channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    self.channels[i - 1],
                    self.channels[i],
                    res2net_scale=self.res2net_scale,
                    se_channels=self.se_channels,
                    kernel_size=self.kernel_sizes[i],
                    dilation=self.dilations[i],
                    groups=self.groups[i],
                ))

        self.mfa = TDNNBlock(
            self.channels[-1],
            self.channels[-1],
            self.kernel_sizes[-1],
            self.dilations[-1],
            self.groups[-1],
        )

        self.asp = AttentiveStatisticsPooling(
            self.channels[-1],
            attention_channels=self.attention_channels,
            global_context=self.global_context,
        )
        self.asp_bn = nn.BatchNorm1d(self.channels[-1] * 2)

        last_in_channels = (self.channels[-1] *
                            2 if self.pooling else self.channels[-1])

        self.fc = Conv1dSamePaddingReflect(in_channels=last_in_channels,
                                           out_channels=self.out_dim,
                                           kernel_size=1)

    @torch.jit.ignore
    def forward(self,
                X: torch.Tensor,
                return_full: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        #Z = super().forward(X)
        Z = X

        feats = []
        for layer in self.blocks:
            Z = layer(Z)
            feats.append(Z)

        Z = torch.cat(feats[1:], dim=1)

        Z = self.mfa(Z)

        # Attentive Statistical Pooling
        if self.pooling:
            Z = self.asp(Z)
            Z = self.asp_bn(Z)

        # Final linear transformation
        Z = self.fc(Z)

        if self.pooling:
            Z = Z.squeeze(dim=2)

        if self.use_tanh:
            Z = torch.tanh(Z)

        if self.spherical_normalisation:
            Z = Z / torch.norm(Z, dim=-1, keepdim=True)

        if self.ac_regularisation:
            kl = (torch.nn.functional.relu((abs(Z) - 1))).mean()
            mean = Z
        else:
            mean = Z
            kl = torch.tensor(0.0).to(Z.device)

        if return_full:
            return Z, mean, kl
        return Z

    @torch.jit.export
    def forward_stream(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        #Z = super().forward(X)
        Z = X

        feats = []
        for layer in self.blocks:
            Z = layer(Z)
            feats.append(Z)

        Z = torch.cat(feats[1:], dim=1)

        Z = self.mfa(Z)

        # Attentive Statistical Pooling
        if self.pooling:
            Z = self.asp(Z)
            Z = self.asp_bn(Z)

        # Final linear transformation
        Z = self.fc(Z)

        if self.pooling:
            Z = Z.squeeze(dim=2)

        #

        if self.use_tanh:
            Z = torch.tanh(Z)

        if self.spherical_normalisation:
            Z = Z / torch.norm(Z, dim=-1, keepdim=True)

        return Z


if __name__ == "__main__":
    # Example usage
    model = ECAPATDNN(
        in_size=80,
        channels=[256, 512, 512, 1024, 2],
        kernel_size=3,
        res2net_scale=8,
        se_channels=128,
        attention_channels=128,
        global_context=True,
        pooling=True,
        use_tanh=True,
        spherical_normalisation=False,
        ac_regularisation=True,
    )

    # Dummy input
    x = torch.randn(16, 80, 100)  # (batch_size, num_features, sequence_length)
    output = model(x, return_full=True)

    ts = torch.jit.script(model)
    print(output)  # Expected output shape
