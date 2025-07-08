from typing import Callable, Optional
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gin
from torch_ema import ExponentialMovingAverage
import os

from einops import reduce, rearrange
import warnings


@gin.configurable
class Base(nn.Module):

    def __init__(self,
                 net,
                 sr,
                 encoder=None,
                 encoder_time=None,
                 post_encoder=None,
                 classifier=None,
                 emb_model=None,
                 time_transform=None,
                 drop_value=-4.,
                 drop_rate=0.2,
                 device="cpu",
                 **kwargs):
        super().__init__()

        self.net = net
        self.encoder = encoder
        self.encoder_time = encoder_time
        self.post_encoder = post_encoder
        self.classifier = classifier

        self.time_transform = time_transform
        self.sr = sr

        self.drop_value = drop_value
        self.drop_rate = drop_rate

        self.extra_modules = nn.ModuleDict({}).to(self.device)

        self.to(device)

        self.emb_model = emb_model

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(self,
               x0: torch.Tensor,
               nb_step: int,
               time_cond=None,
               cond=None,
               guidance: float = 1):
        pass

    def diffusion_step(self, x1, time_cond, cond):
        pass

    def cycle_step(self, interpolant, t, time_cond, cond, cycle_mode,
                   cycle_swap_target, cycle_loss_type, cycle_scaling):
        pass

    def cfgdrop(self, datas, bsize, drop_targets=[], drop_rate=0.2):
        draw = torch.rand(bsize)
        test_drop_all = (draw < drop_rate)

        for i in range(len(datas)):
            test_drop_i = (draw > drop_rate * (i + 1)) & (draw < drop_rate *
                                                          (i + 2))

            test_drop = (test_drop_all +
                         test_drop_i) if i in drop_targets else test_drop_all
            anti_test_drop = ~test_drop

            test_drop = self.broadcast_to(test_drop.to(datas[i]),
                                          datas[i].shape)
            anti_test_drop = self.broadcast_to(anti_test_drop.to(datas[i]),
                                               datas[i].shape)

            if datas[i] is None:
                datas[i] = None
            else:
                datas[i] = anti_test_drop * datas[
                    i] + test_drop * torch.ones_like(
                        datas[i]) * self.drop_value

        return datas

    def broadcast_to(self, alpha, shape):
        assert type(shape) == torch.Size
        return alpha.reshape(-1, *((1, ) * (len(shape) - 1)))

    def init_train(self, lr, dataloader):
        params = list(self.net.parameters())

        if self.encoder is not None:
            print("training encoder")
            params += list(self.encoder.parameters())

        if self.encoder_time is not None:
            print("training encoder_time")
            params += list(self.encoder_time.parameters())

        if self.classifier is not None:
            self.opt_classifier = AdamW(self.classifier.parameters(),
                                        lr=lr,
                                        betas=(0.9, 0.999))

        for k, v in self.extra_modules.items():
            params += list(v.parameters())

        self.opt = AdamW(params, lr=lr, betas=(0.9, 0.999))
        self.step = 0

    @gin.configurable
    @torch.no_grad()
    def prep_data(self, batch, device=None):

        x1 = batch["x"].to(device)
        x1_cond = batch.get("x_cond", x1)
        x1_time_cond = batch.get("x_time_cond", x1)

        x1_cond = x1_cond.to(device)
        x1_time_cond = x1_time_cond.to(device)

        if self.time_transform is not None:
            x1_time_cond = self.time_transform(x1_time_cond)

        return x1, x1_cond, x1_time_cond

    def sample_prior(self, x0_shape):
        return torch.randn(x0_shape).to(self.device)

    def save_model(self, model_dir):
        if self.use_ema:
            with self.ema.average_parameters():
                state_dict = self.state_dict()
                state_dict = {
                    k: v
                    for k, v in state_dict.items() if "emb_model" not in k
                }
                d = {
                    "model_state": {
                        k: v
                        for k, v in state_dict.items() if "emb_model" not in k
                    },
                    "opt_state": self.opt.state_dict()
                }

                torch.save(
                    d, model_dir + "/checkpoint" + str(self.step) + "_EMA.pt")
        else:
            state_dict = self.state_dict()
            state_dict = {
                k: v
                for k, v in state_dict.items() if "emb_model" not in k
            }
            d = {
                "model_state": {
                    k: v
                    for k, v in state_dict.items if "emb_model" not in k
                },
                "opt_state": self.opt.state_dict()
            }

            torch.save(d, model_dir + "/checkpoint" + str(self.step) + ".pt")

    @gin.configurable
    def fit(self,
            dataloader,
            validloader,
            restart_step,
            model_dir,
            max_steps,
            lr,
            adversarial_weight=0.0,
            adversarial_warmup=10000,
            adversarial_loss="cosine",
            timbre_warmup=100000,
            stop_training_encoder_step=1e10,
            stop_training_encoder_time_step=1e10,
            shuffle_zsem=False,
            cycle_start_step=1e10,
            cycle_consistency=False,
            cycle_weights=[0., 0.],
            cycle_mode="interpolant",
            cycle_loss_type="cosine",
            cycle_swap_target="cond",
            cycle_scaling=False,
            regularisation_weight=0.0,
            regularisation_warmup=50000,
            drop_targets="both",
            steps_valid=5000,
            steps_display=100,
            steps_save=25000,
            train_encoder=True,
            train_encoder_time=True,
            use_ema=True,
            update_classifier_every=2,
            load_encoders=[True, True, True],
            zsem_noise_aug=0.,
            time_cond_noise_aug=0.):

        self.train_encoder = train_encoder
        self.train_encoder_time = train_encoder_time
        self.use_ema = use_ema
        self.max_steps = max_steps

        self.init_train(lr=lr, dataloader=validloader)

        if restart_step is not None and restart_step > 0:
            state_dict = torch.load(f"{model_dir}/checkpoint" +
                                    str(restart_step) + "_EMA.pt",
                                    map_location="cpu")

            state_dict_model = {
                key: value
                for key, value in state_dict["model_state"].items()
                if (load_encoders[0] or "encoder." not in key) and (
                    load_encoders[1] or "encoder_time" not in key)
            }

            state_dict_model = {
                key: value
                for key, value in state_dict_model.items()
                if (load_encoders[2] or "net." not in key)
            }

            self.load_state_dict(state_dict_model, strict=False)

            try:
                self.opt.load_state_dict(state_dict["opt_state"])
            except Exception as e:
                print(e)
                print("Could not load optimizer state")
            self.step = restart_step + 1

            print("Restarting from step ", self.step)

        if self.use_ema:
            params = list(self.net.parameters())
            self.ema = ExponentialMovingAverage(params, decay=0.999)

        # Loging
        logger = SummaryWriter(log_dir=model_dir + "/logs")
        self.tepoch = tqdm(total=max_steps, unit="batch")

        n_epochs = max_steps // len(dataloader) + 1
        if restart_step is not None:
            n_epochs = n_epochs - restart_step // len(dataloader)
        losses_sum = {}
        losses_sum_count = {}

        with open(os.path.join(model_dir, "config.gin"), "w") as config_out:
            config_out.write(gin.operative_config_str())

        for e in range(n_epochs):
            for batch in dataloader:
                if (self.step > stop_training_encoder_step
                        and self.train_encoder == True):
                    print("detaching encoder")
                    for param in self.encoder.parameters():
                        param.requires_grad = False
                    self.encoder.eval()
                    self.train_encoder = False

                if (self.step > stop_training_encoder_time_step
                        and self.train_encoder_time == True):
                    print("detaching encoder")
                    for param in self.encoder_time.parameters():
                        param.requires_grad = False
                    self.encoder_time.eval()
                    self.train_encoder_time = False

                x1, x1_cond, x1_time_cond = self.prep_data(batch,
                                                           device=self.device)

                if shuffle_zsem is not None:
                    for n in range(x1_cond.shape[0]):
                        split_size = int(np.random.choice(shuffle_zsem, 1))
                        if split_size == 0:
                            continue
                        else:
                            zsplit = x1_cond[n].split(split_size, dim=-1)
                            zsplit = [
                                zsplit[i] for i in torch.randperm(len(zsplit))
                            ]
                            x1_cond[n] = torch.cat(zsplit, dim=-1)

                if self.step > stop_training_encoder_step or not train_encoder:
                    with torch.no_grad():
                        cond, cond_mean, cond_reg = self.encoder(
                            x1_cond, return_full=True)
                else:
                    cond, cond_mean, cond_reg = self.encoder(x1_cond,
                                                             return_full=True)

                cond = cond + zsem_noise_aug * torch.randn_like(cond)

                if self.encoder_time is not None:
                    if self.step < timbre_warmup:
                        with torch.no_grad():
                            time_cond, time_cond_mean, time_cond_reg = self.encoder_time(
                                x1_time_cond, return_full=True)
                            time_cond = self.drop_value * torch.ones_like(
                                time_cond)
                            time_cond_reg = torch.tensor(0.)
                    else:
                        time_cond, time_cond_mean, time_cond_reg = self.encoder_time(
                            x1_time_cond, return_full=True)
                else:
                    time_cond = x1_time_cond
                    time_cond_reg = torch.tensor(0.)

                time_cond = time_cond + time_cond_noise_aug * torch.randn_like(
                    time_cond)

                if self.drop_rate > 0:
                    if self.step < timbre_warmup:
                        drop_targets = []
                    else:
                        drop_targets = drop_targets

                    cond_drop, time_cond_drop = self.cfgdrop(
                        [cond, time_cond],
                        bsize=x1.shape[0],
                        drop_targets=drop_targets,
                        drop_rate=self.drop_rate)

                # Adversarial step
                if self.step > timbre_warmup and not (
                        self.step % update_classifier_every
                        == 0) and self.classifier is not None:

                    cond_pred = self.classifier(time_cond.detach())

                    if adversarial_loss == "cosine":
                        classifier_loss = (
                            1 - torch.nn.functional.cosine_similarity(
                                cond_pred, cond.detach(), dim=1,
                                eps=1e-8)).mean()

                    elif adversarial_loss == "mse":
                        classifier_loss = torch.nn.functional.mse_loss(
                            cond_pred, cond_mean.detach(), reduction='mean')

                    self.opt_classifier.zero_grad()
                    classifier_loss.backward()

                    self.opt_classifier.step()

                    lossdict = {
                        "Classifier loss": classifier_loss.item(),
                    }

                # Diffusion step
                else:
                    if self.step < timbre_warmup:
                        time_cond_drop = self.drop_value * torch.ones_like(
                            time_cond_drop)

                    if self.step > timbre_warmup and self.classifier is not None:
                        cond_pred = self.classifier(time_cond)
                        if adversarial_loss == "cosine":
                            classifier_loss = (
                                1 - torch.nn.functional.cosine_similarity(
                                    cond_pred, cond.detach(), dim=1,
                                    eps=1e-8)).mean()

                        elif adversarial_loss == "mse":
                            classifier_loss = torch.nn.functional.mse_loss(
                                cond_pred, cond.detach(), reduction='mean')

                    else:
                        classifier_loss = torch.tensor(0.)
                        adversarial_weight_cur = 0.

                    diffusion_loss, interpolant, t = self.diffusion_step(
                        x1, time_cond=time_cond_drop, cond=cond_drop)

                    if cycle_consistency and self.step > cycle_start_step:
                        cond_cycle_loss, time_cond_cycle_loss = self.cycle_step(
                            interpolant, t, time_cond, cond, cycle_mode,
                            cycle_swap_target, cycle_loss_type, cycle_scaling)

                    else:
                        cond_cycle_loss = torch.tensor(0.)
                        time_cond_cycle_loss = torch.tensor(0.)

                    # Compute weights
                    adversarial_weight_cur = min(
                        adversarial_weight * (self.step - timbre_warmup) /
                        (adversarial_warmup), adversarial_weight)

                    regularisation_weight_cur = min(
                        regularisation_weight * self.step /
                        (regularisation_warmup), regularisation_weight)

                    # log losses
                    cycle_weights_cur = cycle_weights if self.step > cycle_start_step else [
                        0., 0.
                    ]

                    lossdict = {
                        "Diffusion loss": diffusion_loss.item(),
                        "Classifier loss": classifier_loss.item(),
                        "Adversarial Regularisation weight":
                        adversarial_weight_cur,
                        "Latent Regularisation weight":
                        regularisation_weight_cur,
                        "Cycle loss - cond": cond_cycle_loss.item(),
                        "Cycle loss - time_cond": time_cond_cycle_loss.item(),
                        "Cycle weight - cond": cycle_weights_cur[0],
                        "Cycle weight - time_cond": cycle_weights_cur[1],
                        "cond_reg": cond_reg.item(),
                        "time_cond_reg": time_cond_reg.item(),
                    }

                    loss = diffusion_loss - adversarial_weight_cur * classifier_loss + cycle_weights_cur[
                        0] * cond_cycle_loss + cycle_weights_cur[
                            1] * time_cond_cycle_loss + regularisation_weight_cur * cond_reg.mean(
                            ) + regularisation_weight_cur * time_cond_reg.mean(
                            )

                    self.opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
                    self.opt.step()

                if self.use_ema:
                    self.ema.update()

                for k in lossdict:
                    losses_sum[k] = losses_sum.get(k, 0.) + lossdict[k]
                    losses_sum_count[k] = losses_sum_count.get(k, 0) + 1

                if self.step % steps_display == 0:
                    self.tepoch.set_postfix(loss=losses_sum["Diffusion loss"] /
                                            steps_display)
                    for k in losses_sum:
                        logger.add_scalar('Loss/' + k,
                                          losses_sum[k] /
                                          max(1, losses_sum_count[k]),
                                          global_step=self.step)
                        losses_sum[k] = 0.
                        losses_sum_count[k] = 0

                if self.step % steps_valid == 20 and validloader is not None:
                    with torch.no_grad() and self.ema.average_parameters():
                        ## VALIDATION

                        lossval = {}

                        for i, batch in enumerate(validloader):
                            x1, x1_cond, x1_time_cond = self.prep_data(
                                batch, device=self.device)

                            cond = self.encoder(x1_cond)
                            time_cond = self.encoder_time(
                                x1_time_cond
                            ) if self.encoder_time is not None else x1_time_cond

                            if self.step < timbre_warmup:
                                time_cond = self.drop_value * torch.ones_like(
                                    time_cond)

                            diffusion_loss, _, _ = self.diffusion_step(
                                x1, time_cond=time_cond, cond=cond)

                            lossdict = {
                                "Diffusion loss": diffusion_loss.item(),
                            }

                            if self.classifier is not None:
                                cond_pred = self.classifier(time_cond)
                                if adversarial_loss == "cosine":
                                    classifier_loss = (
                                        1 -
                                        torch.nn.functional.cosine_similarity(
                                            cond_pred, cond, dim=1,
                                            eps=1e-8)).mean()

                                elif adversarial_loss == "mse":
                                    classifier_loss = torch.nn.functional.mse_loss(
                                        cond_pred, cond, reduction='mean')

                                lossdict[
                                    "Classifier loss"] = classifier_loss.item(
                                    )

                            for k in lossdict:
                                lossval[k] = lossval.get(k, 0.) + lossdict[k]

                            if i == 100:
                                break

                        for k in lossval:
                            logger.add_scalar('Loss/valid/' + k,
                                              lossval[k] / 100,
                                              global_step=self.step)

                        ## SAMPLING
                        x1 = x1[:6].to(self.device)
                        time_cond = time_cond[:6] if time_cond is not None else None
                        cond = cond[:6] if cond is not None else None
                        x0 = self.sample_prior(x1.shape)

                        audio_true = self.emb_model.decode(x1.cpu()).cpu()

                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            for i in range(x1.shape[0]):
                                logger.add_audio("true/" + str(i),
                                                 audio_true[i],
                                                 global_step=self.step,
                                                 sample_rate=self.sr)

                            for nb_steps in [5, 40]:
                                x1_rec = self.sample(x0,
                                                     nb_steps=nb_steps,
                                                     time_cond=time_cond,
                                                     cond=cond)

                                audio_rec = self.emb_model.decode(
                                    x1_rec.cpu()).cpu()

                                for i in range(x1.shape[0]):
                                    logger.add_audio("generated/" +
                                                     str(nb_steps) + "steps/" +
                                                     str(i),
                                                     audio_rec[i],
                                                     global_step=self.step,
                                                     sample_rate=self.sr)

                if self.step % steps_save == 0:
                    self.save_model(model_dir)

                self.tepoch.update(1)
                self.step += 1


class RectifiedFlow(Base):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def smooth_function_cond(self, x, slope=7):
        return 0.5 * (1 + torch.tanh(slope * (0.4 - x)))

    def cycle_step(self,
                   interpolant,
                   t,
                   time_cond,
                   cond,
                   cycle_mode="interpolant",
                   cycle_swap_target="cond",
                   cycle_loss_type="cosine",
                   cycle_scaling=False):

        if cycle_swap_target == "time_cond":
            time_cond_target = time_cond[torch.randperm(time_cond.shape[0])]
            cond_target = cond

        elif cycle_swap_target == "cond":
            permutation = torch.randperm(cond.shape[0])
            time_cond_target = time_cond
            cond_target = cond[permutation]

        elif cycle_swap_target == "alternate":
            n = time_cond.shape[0]
            indices = torch.randperm(time_cond.shape[0])
            time_cond_target = time_cond.clone()
            time_cond_target[indices[:n // 2]] = time_cond[indices[n // 2:]]
            cond_target = cond.clone()
            cond_target[indices[n // 2:]] = cond[indices[:n // 2]]

        time_cond_target = time_cond_target.detach()
        cond_target = cond_target.detach()

        if cycle_mode == "interpolant":
            model_output_transfer = self.net(interpolant,
                                             time=t,
                                             time_cond=time_cond_target,
                                             cond=cond_target)
            x_transfer = interpolant + (1 - t) * model_output_transfer

            cond_rec = self.encoder(x_transfer)
            time_cond_rec = self.encoder_time(x_transfer)

        elif cycle_mode == "sample":
            x0 = torch.randn_like(interpolant)
            with torch.no_grad():
                x_transfer_onestep = self.sample(x0,
                                                 cond_target,
                                                 time_cond_target,
                                                 nb_steps=2,
                                                 guidance_cond_factor=1.,
                                                 guidance_joint_factor=1.,
                                                 total_guidance=1.).detach()

            interpolant = (1 - t) * x0 + t * x_transfer_onestep
            model_output_transfer = self.net(interpolant,
                                             time=t,
                                             time_cond=time_cond_target,
                                             cond=cond_target)
            x_transfer = interpolant + (1 - t) * model_output_transfer
            cond_rec = self.encoder(x_transfer)
            time_cond_rec = self.encoder_time(x_transfer)

        if cycle_loss_type == "mse":
            cond_cycle_loss = torch.nn.functional.mse_loss(
                cond_rec, cond_target.detach(), reduction='none')

            time_cond_cycle_loss = torch.nn.functional.mse_loss(
                time_cond_rec, time_cond_target, reduction="none")

        elif "mse_margin" in cycle_loss_type:
            margin = float(cycle_loss_type.split("_")[-1])
            cond_cycle_loss = torch.maximum(
                torch.tensor(margin),
                torch.nn.functional.mse_loss(cond_rec,
                                             cond_target.detach(),
                                             reduction='none'))

            time_cond_cycle_loss = torch.maximum(
                torch.tensor(margin),
                torch.nn.functional.mse_loss(time_cond_rec,
                                             time_cond_target,
                                             reduction="none"))

        elif cycle_loss_type == "cosine":
            cond_cycle_loss = (1 - torch.nn.functional.cosine_similarity(
                cond_rec, cond_target.detach(), dim=1, eps=1e-8)).mean()

            time_cond_cycle_loss = (1 - torch.nn.functional.cosine_similarity(
                time_cond_rec, time_cond_target.detach(), dim=1,
                eps=1e-8)).mean()
        else:
            raise ValueError("Invalid cycle loss type : " + cycle_loss_type)

        if cycle_scaling == "natural":
            with torch.no_grad():
                model_output_nocond = self.net(interpolant,
                                               time_cond=time_cond,
                                               cond=self.drop_value *
                                               torch.ones_like(cond),
                                               time=t)
                model_output = self.net(interpolant,
                                        time_cond=time_cond,
                                        cond=cond,
                                        time=t)
            scaling_cond = torch.nn.functional.mse_loss(
                model_output.detach(),
                model_output_nocond.detach(),
                reduction="none").mean((1, 2))
            scaling_cond[t.squeeze() > 0.6] = torch.zeros_like(
                scaling_cond[t.squeeze() > 0.6])

            cond_cycle_loss = scaling_cond[:, None] * cond_cycle_loss

        elif cycle_scaling == "ramps":
            scaling_cond = self.smooth_function_cond(t.view(-1))
            cond_cycle_loss = scaling_cond[:, None] * cond_cycle_loss
        elif cycle_scaling == "none":
            pass
        else:
            raise ValueError("Invalid cycle scaling : " + cycle_scaling)

        cond_cycle_loss = cond_cycle_loss.mean()
        time_cond_cycle_loss = time_cond_cycle_loss.mean()

        return cond_cycle_loss, time_cond_cycle_loss

    def diffusion_step(self, x1, time_cond, cond):

        x0 = torch.randn_like(x1)

        target = x1 - x0

        t = torch.rand(x0.size(0), 1, 1).to(self.device)

        interpolant = (1 - t) * x0 + t * x1

        model_output = self.net(interpolant,
                                time_cond=time_cond,
                                cond=cond,
                                time=t)

        loss = ((model_output - target)**2).mean()

        return loss, interpolant, t

    def model_forward(self,
                      x: torch.Tensor,
                      time: torch.Tensor,
                      cond: torch.Tensor,
                      time_cond: torch.Tensor,
                      guidance_timbre: float,
                      guidance_structure: float,
                      cache_index: int = 0) -> torch.Tensor:

        full_time = time.repeat(3, 1, 1)
        full_x = x.repeat(3, 1, 1)

        full_cond = torch.cat([
            cond,
            self.drop_value * torch.ones_like(cond),
            self.drop_value * torch.ones_like(cond),
        ])

        full_time_cond = torch.cat([
            time_cond,
            time_cond,
            self.drop_value * torch.ones_like(time_cond),
        ])

        dx = self.net(full_x,
                      time=full_time,
                      cond=full_cond,
                      time_cond=full_time_cond,
                      cache_index=cache_index)

        dx_full, dx_time_cond, dx_none = torch.chunk(dx, 3, dim=0)

        total_guidance = 0.5 * (guidance_structure + guidance_timbre)

        guidance_cond_factor = guidance_timbre / (max(guidance_structure,
                                                      0.01))

        dx = dx_none + total_guidance * (dx_time_cond + guidance_cond_factor *
                                         (dx_full - dx_time_cond) - dx_none)

        return dx

    @torch.no_grad()
    def sample(self,
               x0,
               cond,
               time_cond,
               nb_steps,
               guidance_timbre=1.,
               guidance_structure=1.):
        dt = 1 / nb_steps
        t_values = torch.linspace(0, 1, nb_steps + 1).to(self.device)[:-1]
        x = x0.to(self.device)

        for t in t_values:
            t = t.reshape(1, 1, 1).repeat(x.shape[0], 1, 1)
            x = x + self.model_forward(
                x=x,
                time=t,
                cond=cond,
                time_cond=time_cond,
                guidance_timbre=guidance_timbre,
                guidance_structure=guidance_structure) * dt

        return x
