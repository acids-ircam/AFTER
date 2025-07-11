import torch
import numpy as np
from random import random
import cached_conv as cc
import gin
import os

from after.autoencoder import AutoEncoder, Trainer
from after.dataset import SimpleDataset, CombinedDataset, random_phase_mangle
from accelerate import Accelerator

from absl import app, flags

FLAGS = flags.FLAGS

cc.use_cached_conv(False)

flags.DEFINE_string("name", "test", "Name of the model")
flags.DEFINE_string("save_dir", "autoencoder_runs",
                    "Log and checkpoint saving directory")
flags.DEFINE_multi_string(
    "db_path", None, "Database path. Use multiple for combined datasets.")
flags.DEFINE_multi_float("freqs", None,
                         "Sampling frequencies for multiple datasets.")
flags.DEFINE_multi_string("config", [], "List of config files")
flags.DEFINE_integer("restart", None, "Restart step")
flags.DEFINE_integer("bsize", 6, "Batch size")
flags.DEFINE_integer("num_signal", 131072, "Number of signals")
flags.DEFINE_integer("gpu", -1, "GPU ID")
flags.DEFINE_integer("num_workers", 0, "Number of workers")
flags.DEFINE_bool("use_cache", False, "Wether to load the dataset in cache")
flags.DEFINE_bool("use_psts", False,
                  "Wether to use the pitch shift and time stretching")


def add_gin_extension(config_name: str) -> str:
    if config_name[-4:] != '.gin':
        config_name += '.gin'
    return config_name


def main(argv):
    model_name = FLAGS.name
    num_signal = FLAGS.num_signal
    step_restart = FLAGS.restart

    device = "cuda:" + str(FLAGS.gpu) if FLAGS.gpu >= 0 else "cpu"

    ## GIN CONFIG
    if FLAGS.restart is not None:
        config_path = os.path.join(FLAGS.save_dir, model_name, "config.gin")
        with gin.unlock_config():
            gin.parse_config_files_and_bindings([config_path], [])

    else:
        gin.parse_config_files_and_bindings(
            map(add_gin_extension, FLAGS.config),
            [],
        )

    sr = gin.query_parameter("%SR")
    #MODELS
    autoencoder = AutoEncoder()

    ### TEST NETWORKS
    x = torch.randn(1, 1, 4096 * 16)
    z, _ = autoencoder.encode(x)
    y = autoencoder.decode(z)

    assert x.shape == y.shape, ValueError(
        f"Shape mismatch: x.shape = {x.shape}, y.shape = {y.shape}")

    ## Start the training
    trainer = Trainer(autoencoder, device=device)

    num_el = 0
    for p in autoencoder.encoder.parameters():
        num_el += p.numel()
    print("Number of parameters - Encoder : ", num_el / 1e6, "M")

    num_el = 0
    for p in autoencoder.decoder.parameters():
        num_el += p.numel()
    print("Number of parameters - Decoder : ", num_el / 1e6, "M")

    num_el = 0
    for p in trainer.discriminator.parameters():
        num_el += p.numel()
    print("Number of parameters - Discriminator : ", num_el / 1e6, "M")

    ## DATASET

    transforms = []

    class RandomApply():
        """
        Apply transform with probability p
        """

        def __init__(self, transform, p=.5):
            self.transform = transform
            self.p = p

        def __call__(self, x: np.ndarray):
            if random() < self.p:
                x = self.transform(x)
            return x

    class RandomGain():

        def __init__(self, db):
            """
            FLAGS:
                db: randomize gain from -db to db. upper bound will be clipped
                    to prevent peak > 1.
            """
            self.db = db

        def __call__(self, x: np.ndarray):
            gain = 10**((random() * (-self.db)) / 20)
            return x * gain

    transforms = [
        RandomApply(lambda x: random_phase_mangle(x, 20, 2000, .99, sr),
                    p=0.8),
        RandomApply(RandomGain(20), p=0.8)
    ]

    if FLAGS.use_psts:
        from after.dataset.transforms import PSTS
        ts = PSTS(sr, ts_min=0.71, ts_max=1.5, pitch_min=-4, pitch_max=+4)
        transforms.append(RandomApply(ts, p=0.5))

    def collate_fn(x):
        x = [l["waveform"] for l in x]
        # x = [
        #     l[..., i0:i0 + num_signal] for l, i0 in zip(
        #         x, torch.randint(x[0].shape[-1] - num_signal, (len(x), )))
        # ]

        for i in range(len(x)):
            x[i] = x[i].reshape(1, -1)
            i0 = np.random.randint(0, x[i].shape[-1] - num_signal)
            x[i] = x[i][..., i0:i0 + num_signal]

        x = np.stack(x)
        for transform in transforms:
            x = transform(x)
        x = torch.from_numpy(x).reshape(x.shape[0], 1, -1).float()
        return x

    if len(FLAGS.db_path) > 1:
        path_dict = {f: {"name": f, "path": f} for f in FLAGS.db_path}

        dataset = CombinedDataset(
            path_dict=path_dict,
            keys=["waveform"],
            freqs="estimate" if FLAGS.freqs is None else FLAGS.freqs,
            config="train",
            init_cache=FLAGS.use_cache,
        )

        train_sampler = dataset.get_sampler()

        valset = CombinedDataset(
            path_dict=path_dict,
            config="validation",
            freqs="estimate" if FLAGS.freqs is None else FLAGS.freqs,
            keys=["waveform"],
            init_cache=FLAGS.use_cache,
        )
        val_sampler = valset.get_sampler()

    else:
        dataset = SimpleDataset(path=FLAGS.db_path[0],
                                keys=["waveform"],
                                init_cache=FLAGS.use_cache,
                                split="train")

        valset = SimpleDataset(path=FLAGS.db_path[0],
                               keys=["waveform"],
                               split="validation",
                               init_cache=FLAGS.use_cache)
        train_sampler, val_sampler = None, None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.bsize,
        shuffle=True if train_sampler is None else False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=FLAGS.num_workers,
        sampler=train_sampler)

    validloader = torch.utils.data.DataLoader(valset,
                                              batch_size=FLAGS.bsize,
                                              shuffle=False,
                                              collate_fn=collate_fn,
                                              drop_last=True,
                                              num_workers=0,
                                              sampler=val_sampler)

    x = next(iter(dataloader))
    print("Training size : ", x.shape)

    if step_restart is not None:
        print("Loading model from step ", step_restart)
        path = os.path.join(FLAGS.save_dir, model_name)
        trainer.load_model(path, step_restart, load_discrim=True)

    accelerator = Accelerator()

    dataloader, _, trainer.model, trainer.discriminator, trainer.opt, trainer.opt_dis = accelerator.prepare(
        dataloader, validloader, trainer.model, trainer.discriminator,
        trainer.opt, trainer.opt_dis)
    trainer.device = accelerator.device

    trainer.fit(dataloader,
                validloader,
                tensorboard=os.path.join(FLAGS.save_dir, model_name),
                accelerator=accelerator)


if __name__ == "__main__":
    app.run(main)
