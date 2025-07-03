import torch
import gin
import numpy as np
from after.dataset import CombinedDataset


def crop(arrays, length, idxs):
    return [
        torch.stack([xc[..., i:i + length] for i, xc in zip(idxs, array)])
        for array in arrays
    ]


def normalize(array):
    return (array - array.min()) / (array.max() - array.min() + 1e-6)


@gin.configurable
def get_datasets(path_dict, data_keys, freqs, use_cache, max_samples):

    dataset = CombinedDataset(
        path_dict=path_dict,
        keys=data_keys,
        freqs="estimate" if freqs is None else freqs,
        config="train",
        init_cache=use_cache,
        num_samples=max_samples,
    )

    train_sampler = dataset.get_sampler()

    valset = CombinedDataset(
        path_dict=path_dict,
        config="validation",
        freqs="estimate" if freqs is None else freqs,
        keys=data_keys,
        init_cache=use_cache,
        num_samples=max_samples,
    )
    val_sampler = valset.get_sampler()
    return dataset, valset, train_sampler, val_sampler


@gin.configurable
def collate_fn(batch,
               n_signal,
               structure_type,
               ae_ratio,
               timbre_limit=None,
               timbre_augmentation_keys=[]):

    x = torch.from_numpy(np.stack([b["z"] for b in batch], axis=0))
    batch_size = x.shape[0]

    i0 = np.random.randint(0, x.shape[-1] - n_signal, x.shape[0])
    x_target = crop([x], n_signal, i0)[0]

    if len(timbre_augmentation_keys) > 0:
        all_timbre, x_timbre = [], []
        for key in timbre_augmentation_keys:
            all_timbre.append([b[key] for b in batch])

        indexes = np.random.randint(0, len(all_timbre), batch_size)
        for i in range(batch_size):
            current_x = all_timbre[indexes[i]][i]

            if current_x.shape[-1] < n_signal + 1:
                current_x = x[i]
                print(
                    "Warning: timbre signal too short, using original signal")
            i1 = np.random.randint(0, current_x.shape[-1] - n_signal, 1)[0]
            current_x = current_x[..., i1:i1 + n_signal]
            x_timbre.append(current_x)

        x_timbre = torch.from_numpy(np.stack(x_timbre, axis=0))

    else:
        if timbre_limit is None:
            i1 = np.random.randint(0, x.shape[-1] - n_signal, x.shape[0])
        else:
            nmax = int(n_signal * timbre_limit)
            i1 = np.random.randint(-nmax, nmax, x.shape[0])
            i1 = [
                np.clip(i0c + i1c, 0, x.shape[-1] - n_signal)
                for i0c, i1c in zip(i0, i1)
            ]
        x_timbre = crop([x], n_signal, i1)[0]

    if structure_type == "audio":
        time_cond_target = x_target
    elif structure_type == "midi":
        midi = [b["midi"] for b in batch]

        times = np.linspace(
            0, x.shape[-1] * ae_ratio / gin.query_parameter("%SR"),
            x.shape[-1])
        pr = [m.get_piano_roll(times=times) for m in midi]

        pr = map(normalize, pr)
        pr = np.stack(list(pr))
        pr = torch.from_numpy(pr).float()
        pr = torch.stack([prc[..., i:i + n_signal] for i, prc in zip(i0, pr)])
        time_cond_target = pr

    return {
        "x": x_target,
        "x_cond": x_timbre,
        "x_time_cond": time_cond_target,
    }
