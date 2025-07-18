import torch
import torch.nn as nn
import cached_conv as cc
import gin


def __prepare_scriptable__(self):
    for hook in self._forward_pre_hooks.values():
        if hook.__module__ == "torch.nn.utils.weight_norm" and hook.__class__.__name__ == "WeightNorm":
            torch.nn.utils.remove_weight_norm(self)
    return self


def normalization(module: nn.Module, mode: str = 'weight_norm'):
    if mode == 'identity':
        return module
    elif mode == 'weight_norm':
        layer = torch.nn.utils.weight_norm(module)
        layer.__prepare_scriptable__ = __prepare_scriptable__.__get__(layer)
        return layer
    else:
        raise Exception(f'Normalization mode {mode} not supported')


@gin.configurable
class V2ConvBlock1D(nn.Module):

    def __init__(self,
                 in_c,
                 out_c,
                 kernel_size,
                 act=nn.SiLU,
                 res=True,
                 cumulative_delay=0,
                 norm_type="batch_norm",
                 num_groups=8):
        super().__init__()
        self.res = res

        conv1 = normalization(
            cc.Conv1d(in_c,
                      out_c,
                      kernel_size=kernel_size,
                      padding=cc.get_padding(kernel_size)))

        conv2 = normalization(
            cc.Conv1d(out_c,
                      out_c,
                      kernel_size=kernel_size,
                      padding=cc.get_padding(kernel_size),
                      cumulative_delay=conv1.cumulative_delay))

        if norm_type == "batch_norm":
            self.gn1 = nn.BatchNorm1d(in_c)
            self.gn2 = nn.BatchNorm1d(out_c)
        elif norm_type == "group_norm":
            self.gn1 = nn.GroupNorm(min(in_c, min(num_groups, in_c//4)), in_c)
            self.gn2 = nn.GroupNorm(min(out_c, min(num_groups, out_c//4)), out_c)
        act = act
        self.dp = nn.Dropout(p=0.1)

        #net = [self.gn1, act(), conv1, self.gn2, act(), self.dp, conv2]
        net = [self.gn1, act(), conv1, self.gn2, act(), self.dp, conv2]
        net = cc.CachedSequential(*net)

        additional_delay = net.cumulative_delay

        self.net = cc.AlignBranches(
            net,
            nn.Identity(),
            delays=[additional_delay, 0],
        )
        self.cumulative_delay = additional_delay + cumulative_delay

    def forward(self, x):
        x, res = self.net(x)
        return x + res


@gin.configurable
class V2EncoderBlock1D(nn.Module):

    def __init__(self,
                 in_c,
                 tot_cond_channels,
                 out_c,
                 kernel_size,
                 ratio,
                 cumulative_delay=0,
                 norm_type="batch_norm"):
        super().__init__()
        conv = V2ConvBlock1D(in_c + tot_cond_channels,
                             in_c,
                             kernel_size,
                             cumulative_delay=cumulative_delay,
                             norm_type=norm_type)

        if ratio != 1:
            pool = normalization(
                cc.Conv1d(in_c,
                          out_c,
                          kernel_size=2 * ratio,
                          stride=ratio,
                          padding=cc.get_padding(2 * ratio, ratio),
                          cumulative_delay=conv.cumulative_delay))
        else:
            pool = normalization(
                cc.Conv1d(in_c,
                          out_c,
                          kernel_size=1,
                          stride=ratio,
                          padding=(0, 0),
                          cumulative_delay=conv.cumulative_delay))

        net = [conv, pool]

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net[-1].cumulative_delay

    def forward(self, x):
        return self.net(x)


@gin.configurable
class Encoder1D(nn.Module):

    def __init__(self,
                 in_size=1,
                 channels=[64, 128, 128, 256, 256],
                 ratios=[2, 2, 2, 2, 2],
                 kernel_size=5,
                 cond={},
                 use_tanh=True,
                 average_out=False,
                 upscale_out=False,
                 vector_quantizer=None,
                 spherical_normalization=False,
                 vae_regularisation=False,
                 ac_regularisation=False,
                 wassertstein_regularisation=False,
                 norm_type="batch_norm"):
        super().__init__()

        self.use_tanh = use_tanh
        self.channels = channels
        self.average_out = average_out
        self.vector_quantizer = vector_quantizer
        self.spherical_normalization = spherical_normalization
        self.vae_regularisation = vae_regularisation
        self.ac_regularisation = ac_regularisation
        self.wassertstein_regularisation = wassertstein_regularisation

        ratios = [1] + ratios
        n = len(self.channels)

        self.out_channels = channels[-1]

        self.cond_modules = nn.ModuleDict()
        self.cond_keys = list(cond.keys())

        c_channels = 0
        for key, p in cond.items():
            if p["num_classes"] == 0:
                self.cond_modules[key] = nn.Sequential(
                    nn.Linear(1, p["emb_dim"]), nn.ReLU())

            else:
                self.cond_modules[key] = nn.Embedding(p["num_classes"] + 1,
                                                      p["emb_dim"])
            c_channels += p["emb_dim"]

        net = []

        if upscale_out:
            channels[-1] = channels[-1] * 4

        if self.vae_regularisation:
            channels[-1] = channels[-1] * 2

        net.append(
            V2EncoderBlock1D(in_size,
                             c_channels,
                             channels[0],
                             kernel_size,
                             ratio=ratios[0],
                             norm_type=norm_type))

        for i in range(1, n):
            net.append(
                V2EncoderBlock1D(channels[i - 1],
                                 c_channels,
                                 channels[i],
                                 kernel_size,
                                 ratios[i],
                                 cumulative_delay=net[-1].cumulative_delay,
                                 norm_type=norm_type))

        net.append(
            V2ConvBlock1D(channels[-1] + c_channels,
                          channels[-1],
                          kernel_size,
                          cumulative_delay=net[-1].cumulative_delay,
                          norm_type=norm_type))

        self.net = cc.CachedSequential(*net)

        total_ratio = 1
        for r in ratios:
            total_ratio *= r
        self.total_ratio = int(total_ratio)

        self.upscale_out = upscale_out
        cumulative_delay = self.net.cumulative_delay

        if self.upscale_out:
            self.upscaler = []
            for i in range(self.total_ratio // 2):
                self.upscaler.append(
                    cc.ConvTranspose1d(
                        in_channels=channels[-1],
                        out_channels=channels[-1],
                        kernel_size=2 * 2,
                        stride=2,
                        padding=2 // 2,
                        cumulative_delay=self.upscaler[-1].cumulative_delay
                        if i > 0 else cumulative_delay)),

                self.upscaler.append(
                    cc.Conv1d(
                        channels[-1],
                        channels[-1],
                        kernel_size=kernel_size,
                        padding=cc.get_padding(kernel_size),
                        cumulative_delay=self.upscaler[-1].cumulative_delay))

            self.upscaler.append(
                cc.Conv1d(channels[-1],
                          channels[-1] // 4,
                          kernel_size,
                          cumulative_delay=self.upscaler[-1].cumulative_delay,
                          padding=cc.get_padding(kernel_size)))

            self.upscaler = cc.CachedSequential(*self.upscaler)
            self.cumulative_delay = self.upscaler[-1].cumulative_delay

        else:
            self.upscaler = nn.Identity()
            self.cumulative_delay = cumulative_delay

    def compute_mean_kernel(self, x, y):
        kernel_input = (x[:, None] - y[None]).pow(2).mean(2) / x.shape[-1]
        return torch.exp(-kernel_input).mean()

    def compute_mmd(self, x, y):
        x_kernel = self.compute_mean_kernel(x, x)
        y_kernel = self.compute_mean_kernel(y, y)
        xy_kernel = self.compute_mean_kernel(x, y)
        mmd = x_kernel + y_kernel - 2 * xy_kernel
        return mmd

    def reparametrize(self, z):
        if self.vae_regularisation:
            mean, scale = z.chunk(2, 1)

            std = nn.functional.softplus(scale) + 1e-4
            var = std * std
            #print(mean)
            #print(std)
            logvar = torch.log(var)

            z = torch.randn_like(mean) * std + mean
            kl = (mean * mean + var - logvar - 1).sum(1).mean()
        elif self.ac_regularisation:
            kl = (torch.nn.functional.relu((abs(z) - 1))).mean()
            mean = z
        elif self.wassertstein_regularisation:
            kl = self.compute_mmd(z, torch.randn_like(z)).mean()
            mean = z
        else:
            kl = torch.tensor(0.).to(z)
            mean = z

        return z, mean, kl

    @torch.jit.ignore
    def forward(self, x, return_full: bool = False):

        x = self.net(x)

        if self.average_out:
            x = torch.mean(x, dim=-1)

        if self.spherical_normalization:
            x = x / (x.norm(dim=1, keepdim=True, p=None) + 1e-5)

        elif self.use_tanh:
            x = torch.tanh(x)

        x, mean, kl = self.reparametrize(x)

        if self.vector_quantizer is not None:
            x, _ = self.vector_quantizer(x)

        if self.upscale_out:
            x = self.upscaler(x)

        if return_full:
            return x, mean, kl
        else:
            return x

    @torch.jit.export
    def forward_stream(self, x):

        x = self.net(x)

        if self.average_out:
            x = torch.mean(x, dim=-1)

        if self.spherical_normalization:
            x = x / (x.norm(dim=1, keepdim=True, p=None) + 1e-5)

        elif self.use_tanh:
            x = torch.tanh(x)

        x, mean, kl = self.reparametrize(x)

        if self.vector_quantizer is not None:
            x, _ = self.vector_quantizer(x)

        if self.upscale_out:
            x = self.upscaler(x)

        return x


def compute_mean_kernel(x, y):
    kernel_input = (x[:, None] - y[None]).pow(2).mean(2) / x.shape[-1]
    return torch.exp(-kernel_input).mean()


def compute_mmd(x, y):
    x_kernel = compute_mean_kernel(x, x)
    y_kernel = compute_mean_kernel(y, y)
    xy_kernel = compute_mean_kernel(x, y)
    mmd = x_kernel + y_kernel - 2 * xy_kernel
    return mmd


@gin.configurable
class LinearEncoder(nn.Module):

    def __init__(self,
                 in_size=512,
                 channels=[512, 1024, 1024, 256, 8],
                 drop_out=0.15,
                 use_tanh=False,
                 regularisation="none"):
        #out_fn=nn.Identity(),
        #**kwargs):
        super().__init__()
        self.use_tanh = use_tanh
        module_list = []
        module_list.append(nn.Linear(in_size, channels[0]))

        if regularisation == "vae":
            channels[-1] = channels[-1] * 2

        for i in range(len(channels) - 1):
            module_list.append(nn.SiLU())
            module_list.append(nn.Dropout(p=drop_out))
            module_list.append(nn.Linear(channels[i], channels[i + 1]))

        self.net = nn.Sequential(*module_list)

        self.regularisation = regularisation

    def calc_expansion(self, z, eps=0.5) -> torch.Tensor:
        """
        Compute expansion loss using Coding Rate estimation.
        """
        m, p = z.shape

        W_centered = z - z.mean(dim=0, keepdim=True)
        cov = (W_centered.T @ W_centered) / (m - 1)

        scalar = p / (m * eps)
        I = torch.eye(p, device=cov.device)
        loss = torch.linalg.cholesky_ex(I + scalar *
                                        cov)[0].diagonal().log().sum()

        loss *= (p * m) / (
            p * m
        )  # the balancing factor gamma, you can also use the next line. This is ultimately a heuristic, so feel free to experiment.
        # loss *= ((self.eps * N * m) ** 0.5 / p)
        return loss

        #self.out_fn = out_fn

    @torch.jit.ignore
    def forward(self, x, return_full=False):
        if self.use_tanh:
            x = torch.tanh(self.net(x))
        else:
            x = self.net(x)

        if self.regularisation == "wasserstein":
            kl = compute_mmd(x, torch.randn_like(x)).mean()
            mean = x

        elif self.regularisation == "vae":
            mean, scale = x.chunk(2, 1)

            std = nn.functional.softplus(scale) + 1e-4
            var = std * std
            logvar = torch.log(var)

            x = torch.randn_like(mean) * std + mean
            kl = (mean * mean + var - logvar - 1).sum(1).mean()
            kl = kl  #* 1e-4

        elif self.regularisation == "ac":
            kl = (1 + torch.nn.functional.relu((abs(x) - 1))).mean()
            mean = x

        # elif self.regularisation == "expansion":
        #     kl = self.calc_expansion(x)
        #     mean = x
        else:
            mean = x
            kl = torch.tensor(0.).to(x)

        if return_full:
            return x, mean, kl

        return x

    @torch.jit.export
    def forward_stream(self, x):
        x = self.net(x)
        if self.use_tanh:
            x = torch.tanh(x)

        if self.regularisation == "vae":
            x, _ = x.chunk(2, 1)

        return x
