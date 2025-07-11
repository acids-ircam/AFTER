from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops_exts import rearrange_many
from torch.optim import AdamW, Adam
from .core import DistanceWrap
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from tqdm import tqdm
import gin
import os


class Dummy():

    def __getattr__(self, key):

        def dummy_func(*args, **kwargs):
            return None

        return dummy_func


@gin.configurable
class Trainer(nn.Module):

    def __init__(self,
                 model: nn.Module,
                 waveform_losses: List[Tuple[int, nn.Module]] = [],
                 reg_losses: List[Tuple[int, nn.Module]] = [],
                 multiband_distances: List[Tuple[int, nn.Module]] = [],
                 sr: int = 16000,
                 max_steps: int = 1000000,
                 discriminator=None,
                 warmup_steps=0,
                 freeze_encoder_step=1000000000,
                 device="cpu",
                 update_discriminator_every: int = 3,
                 accelerator=None):

        super().__init__()

        self.waveform_losses = nn.ModuleList([
            DistanceWrap(scale, loss) for scale, loss in waveform_losses
        ]).to(device)
        self.reg_losses = nn.ModuleList([
            DistanceWrap(scale, loss) for scale, loss in reg_losses
        ]).to(device)
        self.multiband_distances = nn.ModuleList([
            DistanceWrap(scale, loss) for scale, loss in multiband_distances
        ]).to(device) if len(multiband_distances) > 0 else []

        self.model = model.to(device)
        self.discriminator = discriminator.to(device)
        self.sr = sr
        self.max_steps = max_steps
        self.warmup = False
        self.warmup_steps = warmup_steps
        self.freeze_encoder_step = freeze_encoder_step
        self.step = 0
        self.device = device
        self.update_discriminator_every = update_discriminator_every

        self.init_opt()

    def compute_loss(self,
                     x,
                     y,
                     x_multiband=None,
                     y_multiband=None,
                     regloss=None):
        total_loss = 0.

        losses = {}
        for dist in self.waveform_losses:
            loss_value = dist(x, y)
            losses[dist.name] = loss_value.item()
            total_loss += loss_value * dist.scale

        total_loss = total_loss * self.weight_waveform_losses
        if regloss is not None:
            cur_weight = min(self.step / self.warmup_regularisation_loss,
                             1.) * self.weight_regularisation_loss
            total_loss += cur_weight * regloss

        if x_multiband is not None and y_multiband is not None:
            for dist in self.multiband_distances:
                loss_value = dist(x_multiband, y_multiband)
                losses[dist.name + "_multiband"] = loss_value.item()
                total_loss += loss_value * dist.scale

        losses["total_loss"] = torch.tensor(total_loss).item()
        if regloss is not None:
            losses["regularisation_loss"] = regloss.item()

        return total_loss, losses

    def get_losses_names(self, accelerator):
        names = []
        for loss in self.reg_losses + self.waveform_losses:
            names.append(loss.name)
        names.extend(["total_loss"])
        names.extend(["regularisation_loss"])

        if True:  #self.model.pqmf_bands > 1:
            for loss in self.multiband_distances:
                names.append(loss.name + "_multiband")

        if self.discriminator is not None:
            names.extend(
                accelerator.unwrap_model(
                    self.discriminator).get_losses_names())
        self.losses_names = names
        return names

    def init_opt(self, lr=1e-4):
        print("warning, putting all models paramters")

        parameters = list(self.model.encoder.parameters()) + list(
            self.model.decoder.parameters())
        self.opt = AdamW(parameters, lr=lr, betas=(0.9, 0.999))

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt,
                                                                gamma=0.999996)

        if self.discriminator is not None:
            self.opt_dis = AdamW(self.discriminator.parameters(),
                                 lr=lr,
                                 betas=(0.8, 0.9))
            self.scheduler_dis = torch.optim.lr_scheduler.ExponentialLR(
                self.opt, gamma=0.999996)

    def load_model(self, path, step, load_discrim=False):
        d = torch.load(path + "/checkpoint" + str(step) + ".pt")
        self.model.load_state_dict(d["model_state"])
        self.opt.load_state_dict(d["opt_state"])

        if load_discrim == True:
            self.discriminator.load_state_dict(d["dis_state"])
            self.opt_dis.load_state_dict(d["opt_dis_state"])
        self.step = step + 1

    def update_waveform_losses(self, rec_loss_decay):
        if self.step < self.warmup_steps:
            self.weight_waveform_losses = 1.
        else:
            self.weight_waveform_losses = rec_loss_decay**(self.step -
                                                           self.warmup_steps)

    def training_step(self, x, accelerator=None):

        self.train()

        if (self.discriminator is not None and self.warmup
            ) and self.step % self.update_discriminator_every == 0:

            with torch.no_grad():
                y, y_multiband, z, regloss, x_multiband = self.model(
                    x, return_all=True)
                #z, x_multiband, _ = self.model.encode(x, with_multi=True)
                #y, y_multiband = self.model.decode(z, with_multi=True)

            loss_out = {}
            loss_gen, loss_dis, loss_dis_dict = self.discriminator(x, y)

            self.opt_dis.zero_grad()
            if accelerator is not None:
                accelerator.backward(loss_dis)
            else:
                loss_dis.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(),
                                           2.0)
            self.opt_dis.step()
            loss_out.update(loss_dis_dict)

        else:
            # if self.step > self.freeze_encoder_step:
            #     with torch.no_grad():
            #         z, x_multiband, regloss = self.model.encoder(
            #             x, with_multi=True)
            #         z = z.detach()
            # else:
            #     z, x_multiband, regloss = self.model.encoder(x, with_multi=True)

            y, y_multiband, z, regloss, x_multiband = self.model(
                x, return_all=True)

            #y, y_multiband = self.model.decode(z, with_multi=True)
            loss_ae, loss_out = self.compute_loss(x,
                                                  y,
                                                  x_multiband=x_multiband,
                                                  y_multiband=y_multiband,
                                                  regloss=regloss)

            if self.warmup:
                loss_gen, loss_dis, loss_dis_dict = self.discriminator(x, y)
            else:
                loss_gen = 0.
                loss_dis_dict = {
                    k: 0.
                    for k in accelerator.unwrap_model(
                        self.discriminator).get_losses_names()
                }

            loss_gen = loss_gen + loss_ae

            loss_out.update(loss_dis_dict)
            self.opt.zero_grad()
            if accelerator is not None:
                accelerator.backward(loss_gen)
            else:
                loss_gen.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            self.opt.step()

        return loss_out

    def val_step(self, validloader, get_audio=False, get_losses=True):

        tval = tqdm(range(len(validloader)), unit="batch")

        #self.eval()
        all_losses = {key: 0 for key in self.losses_names}

        with torch.no_grad():
            for i, x in enumerate(validloader):
                x = x.to(self.device)
                y, y_multiband, z, regloss, x_multiband = self.model(
                    x, return_all=True)

                _, losses = self.compute_loss(x, y)

                for k, v in losses.items():
                    all_losses[k] += v

                tval.update(1)

                if get_losses == False:
                    break

                if i == 50:
                    break

            all_losses = {
                k: v / min(50, len(validloader))
                for k, v in all_losses.items()
            }
            if get_audio:
                x, y = x.cpu()[:4], y.cpu()[:4]
                audio = torch.cat(
                    (x, torch.zeros(
                        (x.shape[0], x.shape[1], int(self.sr / 3))), y),
                    dim=-1)
                audio = audio.reshape(1, 1, -1)
                if get_losses == False:
                    return audio
                return all_losses, audio
            else:
                return all_losses, None

    @gin.configurable
    def fit(self,
            trainloader,
            validloader,
            tensorboard=None,
            steps_display=20,
            rec_loss_decay=0.999996,
            weight_regularisation_loss=1.,
            warmup_regularisation_loss=100000,
            accelerator=None):

        if tensorboard is not None and accelerator.is_main_process:
            logger = SummaryWriter(log_dir=tensorboard)
        else:
            logger = Dummy()

        tepoch = tqdm(range(self.step, self.max_steps),
                      initial=self.step,
                      unit="batch")
        all_losses_sum = {
            key: 0
            for key in self.get_losses_names(accelerator=accelerator)
        }
        all_losses_count = {
            key: 0
            for key in self.get_losses_names(accelerator=accelerator)
        }
        self.weight_waveform_losses = 1.
        self.weight_regularisation_loss = weight_regularisation_loss
        self.warmup_regularisation_loss = warmup_regularisation_loss

        with open(os.path.join(tensorboard, "config.gin"), "w") as config_out:
            config_out.write(gin.operative_config_str())

        while self.step < self.max_steps:
            for x in trainloader:
                x = x.to(self.device)

                all_losses = self.training_step(x, accelerator=accelerator)

                for k in all_losses:
                    all_losses_sum[k] += all_losses[k]
                    all_losses_count[k] += 1

                if accelerator.is_main_process:
                    tepoch.update(1)

                self.update_waveform_losses(rec_loss_decay)

                if not self.step % steps_display and accelerator.is_main_process:
                    tepoch.set_postfix(loss=all_losses_sum["total_loss"] /
                                       steps_display)
                    for k in all_losses_sum:
                        logger.add_scalar('Loss/' + k,
                                          all_losses_sum[k] /
                                          all_losses_count[k],
                                          global_step=self.step)
                        all_losses_sum[k] = 0.
                        all_losses_count[k] = 0

                if (not self.step % 10000):
                    accelerator.wait_for_everyone()
                    print("Validation Step")

                    if validloader is not None:
                        all_losses, audio = self.val_step(validloader,
                                                          get_audio=True)

                        print("Validation Loss at step ", self.step, " : ",
                              all_losses["total_loss"])
                        #
                        if logger and accelerator.is_main_process:
                            for k, v in all_losses.items():
                                logger.add_scalar('Validation/' + k,
                                                  v,
                                                  global_step=self.step)

                            logger.add_audio("Validation/Audio",
                                             audio,
                                             global_step=self.step,
                                             sample_rate=self.sr)
                if not (self.step % 50000):
                    print("saving:", accelerator.process_index)
                    if accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(self.model)
                        unwrapped_discriminator = accelerator.unwrap_model(
                            self.discriminator)

                        d = {
                            "model_state": unwrapped_model.state_dict(),
                            "opt_state": self.opt.state_dict(),
                            "dis_state": unwrapped_discriminator.state_dict(),
                            "opt_dis_state": self.opt_dis.state_dict()
                        }

                        accelerator.save(
                            d, tensorboard + "/checkpoint" + str(self.step) +
                            ".pt")

                    accelerator.wait_for_everyone()
                    print("finished saving:", accelerator.process_index)

                if self.step > self.max_steps + 1000:
                    exit()

                if self.step > self.warmup_steps and self.warmup == False:
                    self.warmup = True
                    print("Warmup finished")

                self.step += 1
