from typing import Callable, Iterable, Sequence, Tuple
import lmdb
import torch
import numpy as np
from after.dataset import SimpleDataset, AudioExample
from after.utils import resolve_device
from tqdm import tqdm
from absl import app, flags

torch.set_grad_enabled(False)

FLAGS = flags.FLAGS

flags.DEFINE_string('db_path',
                    default=None,
                    help='Database path',
                    required=True)

flags.DEFINE_string('emb_model_path',
                    default=None,
                    help='Autoencoder model path',
                    required=True)

flags.DEFINE_integer('gpu',
                     default=-1,
                     help='Legacy CUDA gpu index for embedding computation. '
                          '--device takes precedence when set.',
                     required=False)
flags.DEFINE_string('device', None,
                    "Torch device: 'cpu', 'cuda', 'cuda:N', 'mps', or 'auto'. "
                    "Overrides --gpu when set.")

flags.DEFINE_integer('batch_size',
                     32,
                     help='batch size for embedding computation',
                     required=False)


def main(dummy):
    device = resolve_device(FLAGS.device, FLAGS.gpu)
    emb_model = torch.jit.load(FLAGS.emb_model_path).to(device)

    bsize = FLAGS.batch_size

    dataset = SimpleDataset(keys=["waveform"],
                            path=FLAGS.db_path,
                            readonly=False)
    env = dataset.env

    total_samples = len(dataset)

    audios = []
    indexes = []

    pbar = tqdm(total=total_samples // bsize)

    for i in range(total_samples):

        audio = dataset[i]["waveform"]
        indexes.append(i)
        audios.append(torch.from_numpy(audio).reshape(1, 1, -1))

        if len(audios) == bsize or i == total_samples - 1:

            emb = emb_model.encode(torch.cat(audios, 0).to(device)).cpu()

            with env.begin(write=True) as txn:
                for idx, z in zip(indexes, emb):
                    print(idx)
                    key = dataset.keys[idx]
                    ae = AudioExample(txn.get(key))
                    ae.put_array("z", z.numpy().squeeze(), dtype=np.float32)
                    txn.put(key, bytes(ae))

            audios = []
            indexes = []
            pbar.update(1)


if __name__ == '__main__':
    app.run(main)
