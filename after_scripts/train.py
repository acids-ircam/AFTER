import gin

#gin.add_config_file_search_path('./after/diffusion/configs')
import torch
import os
import numpy as np

import after
from after.dataset import SimpleDataset, CombinedDataset
from after.diffusion.utils import collate_fn
from tqdm import tqdm

from absl import flags, app

FLAGS = flags.FLAGS

# MODEL
flags.DEFINE_string("name", "test", "Name of the model.")
flags.DEFINE_integer("restart", None, "Restart flag.")
flags.DEFINE_integer("gpu", 0, "GPU ID to use.")
flags.DEFINE_multi_string("config", [], "List of config files.")
flags.DEFINE_string("model", "rectified", "Model type.")

# Training
flags.DEFINE_integer("bsize", 32, "Batch size.")
flags.DEFINE_integer("n_signal", 128,
                     "Training length in number of latent steps")

# DATASET
flags.DEFINE_multi_string(
    "db_path", None, "Database path. Use multiple for combined datasets.")
flags.DEFINE_multi_float("freqs", None,
                         "Sampling frequencies for multiple datasets.")
flags.DEFINE_string("out_path", "./after_runs", "Output path.")
flags.DEFINE_string("emb_model_path", None, "Path to the embedding model.")

# Puts the dataset in cache prior to training for slow hard drives
flags.DEFINE_bool("use_cache", False, "Whether to cache the dataset.")
flags.DEFINE_integer("max_samples", None, "Maximum number of samples.")
flags.DEFINE_integer("num_workers", 0, "Number of workers.")
flags.DEFINE_multi_string("augmentation_keys", ["all"],
                          "List of augmentation keys.")

flags.DEFINE_bool("use_validation", True, "Use a train/validation split")


def add_gin_extension(config_name: str) -> str:
    if config_name[-4:] != '.gin':
        config_name += '.gin'
    return config_name


def main(argv):

    print(FLAGS.config)

    gin.parse_config_files_and_bindings(
        map(add_gin_extension, FLAGS.config),
        [],
    )

    if FLAGS.restart is not None:
        config_path = os.path.join(FLAGS.out_path, FLAGS.name, "config.gin")
        with gin.unlock_config():
            gin.parse_config_files_and_bindings([config_path], [])

    device = "cuda:" + str(FLAGS.gpu) if FLAGS.gpu >= 0 else "cpu"

    ######### BUILD MODEL #########

    emb_model = torch.jit.load(FLAGS.emb_model_path).cpu()  #.to(device)
    dummy = torch.randn(1, 1, 4096)  #.to(device)
    z = emb_model.encode(dummy)
    ae_emb_size = z.shape[1]
    ae_ratio = 4096 // z.shape[-1]

    print("using a codec with - compression ratio : ", ae_ratio,
          " - emb size : ", ae_emb_size)

    with gin.unlock_config():
        gin.bind_parameter("diffusion.utils.collate_fn.ae_ratio", ae_ratio)
        gin.bind_parameter("%IN_SIZE", ae_emb_size)

        if gin.query_parameter("%N_SIGNAL") is None:
            print("setting n_signal with FLAGS")
            gin.bind_parameter("%N_SIGNAL", FLAGS.n_signal)

    if FLAGS.model == "rectified":
        from after.diffusion import RectifiedFlow
        blender = RectifiedFlow(device=device, emb_model=emb_model)
    elif FLAGS.model == "edm":
        from after.diffusion import EDM
        blender = EDM(device=device, emb_model=emb_model)
    else:
        raise ValueError("Model not recognized")

    ######### GET THE DATASET #########
    n_signal = gin.query_parameter("%N_SIGNAL")
    n_signal_waveform = n_signal * ae_ratio
    structure_type = gin.query_parameter("%STRUCTURE_TYPE")

    data_keys = ["z"
                 ] + (["waveform"] if blender.time_transform is not None else
                      []) + (["midi"] if structure_type == "midi" else [])

    ## DATASET
    augmentation_keys = FLAGS.augmentation_keys

    if augmentation_keys == ["all"]:
        dataset = SimpleDataset(path=FLAGS.db_path[0])
        allkeys = dataset.get_keys()
        augmentation_keys = [
            k for k in allkeys if "augment" in k or "aug" in k
        ]

    if augmentation_keys is not None:
        print("Augmentation keys", augmentation_keys)

        with gin.unlock_config():
            gin.bind_parameter(
                "diffusion.utils.collate_fn.timbre_augmentation_keys",
                augmentation_keys)

        data_keys = data_keys + augmentation_keys
    else:
        print("No augmentation keys")

    if len(FLAGS.db_path) > 1:
        path_dict = {f: {"name": f, "path": f} for f in FLAGS.db_path}

        dataset = CombinedDataset(
            path_dict=path_dict,
            keys=data_keys,
            freqs="estimate" if FLAGS.freqs is None else FLAGS.freqs,
            config="train",
            init_cache=FLAGS.use_cache,
            num_samples=FLAGS.max_samples,
        )

        train_sampler = dataset.get_sampler()

        if FLAGS.use_validation:
            valset = CombinedDataset(
                path_dict=path_dict,
                config="validation",
                freqs="estimate" if FLAGS.freqs is None else FLAGS.freqs,
                keys=data_keys,
                init_cache=FLAGS.use_cache,
                num_samples=FLAGS.max_samples,
            )
            val_sampler = valset.get_sampler()

    else:
        dataset = SimpleDataset(path=FLAGS.db_path[0],
                                keys=data_keys,
                                max_samples=FLAGS.max_samples,
                                init_cache=FLAGS.use_cache,
                                split="train")
        if FLAGS.use_validation:
            valset = SimpleDataset(path=FLAGS.db_path[0],
                                   keys=data_keys,
                                   max_samples=FLAGS.max_samples,
                                   split="validation",
                                   init_cache=FLAGS.use_cache)
        train_sampler, val_sampler = None, None

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.bsize,
        shuffle=True if train_sampler is None else False,
        num_workers=FLAGS.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=train_sampler if train_sampler is not None else None)

    if FLAGS.use_validation:
        valid_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=FLAGS.bsize,
            shuffle=False,
            num_workers=FLAGS.num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=val_sampler if val_sampler is not None else None)
    else:
        valid_loader = None

    print("Data shape : ", dataset[0]["z"].shape)
    print("Croped shape : ", next(iter(train_loader))["x"].shape)

    try:
        dummy = collate_fn([])
    except:
        pass

    ######### SAVE CONFIG #########
    model_dir = os.path.join(FLAGS.out_path, FLAGS.name)
    os.makedirs(model_dir, exist_ok=True)

    ######### PRINT NUMBER OF PARAMETERS #########
    num_el = 0
    for p in blender.net.parameters():
        num_el += p.numel()
    print("Number of parameters - unet : ", num_el / 1e6, "M")

    if blender.encoder is not None:
        num_el = 0
        for p in blender.encoder.parameters():
            num_el += p.numel()
        print("Number of parameters - encoder : ", num_el / 1e6, "M")

    if blender.encoder_time is not None:
        num_el = 0
        for p in blender.encoder_time.parameters():
            num_el += p.numel()
        print("Number of parameters - encoder_time : ", num_el / 1e6, "M")

    if blender.classifier is not None:
        num_el = 0
        for p in blender.classifier.parameters():
            num_el += p.numel()
        print("Number of parameters - classifier : ", num_el / 1e6, "M")

    ######### TRAINING #########
    d = {
        "model_dir": model_dir,
        "dataloader": train_loader,
        "validloader": valid_loader,
        "restart_step": FLAGS.restart,
    }

    blender.fit(**d)


if __name__ == "__main__":
    app.run(main)
