from typing import Callable, Iterable, Sequence, Tuple
import pathlib
import librosa
import lmdb
import torch
import numpy as np
from after.dataset.audio_example import AudioExample
from after.dataset.parsers import get_parser
import os
from tqdm import tqdm
from after.dataset.transforms import BasicPitchPytorch, PSTS, AudioDescriptors, BeatTrack
import pickle
import pretty_midi
from absl import app, flags
import copy
from multiprocessing import Pool
from functools import partial

torch.set_grad_enabled(False)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_path',
    None,
    help=
    'Path to a directory containing audio files - use slakh main directory to use slakh',
    required=True)
flags.DEFINE_string('output_path',
                    ".",
                    help='Output directory for the dataset',
                    required=False)

flags.DEFINE_string(
    'midi_path',
    None,
    help='Folder containing the midi files, if a midi file parser is used',
    required=False)

flags.DEFINE_string(
    'parser',
    "simple_audio",
    help=
    'File parser defined in parsers.py. Use None for recursive search of audio files in the input path',
    required=False)

flags.DEFINE_multi_string('exclude', [],
                          help='kewords to exclude from the search',
                          required=False)

flags.DEFINE_multi_string('include',
                          None,
                          help='kewords to include in the file search',
                          required=False)

flags.DEFINE_bool('normalize',
                  True,
                  help='Normalize audio files magnitude',
                  required=False)

flags.DEFINE_bool('cut_silences',
                  True,
                  help='Remove silence chunks',
                  required=False)

flags.DEFINE_integer('num_signal',
                     524288,
                     help='Number of audio samples to use during training')

flags.DEFINE_integer('sample_rate',
                     44100,
                     help='Sampling rate to use during training')

flags.DEFINE_integer('db_size',
                     200,
                     help='Maximum size (in GB) of the dataset')

flags.DEFINE_string(
    'emb_model_path',
    None,
    help='Embedding model path for precomputing the AE embeddings',
    required=False)

flags.DEFINE_integer('batch_size', 8, help='Number of chunks', required=False)
flags.DEFINE_integer('gpu',
                     "-1",
                     help='Cuda gpu index. Use -1 for cpu',
                     required=False)

flags.DEFINE_multi_string(
    'ext',
    default=['wav', 'opus', 'mp3', 'aac', 'flac'],
    help='Extension to search for in the input directory')

flags.DEFINE_bool('save_waveform',
                  default=False,
                  help="Save the waveform in the database")

flags.DEFINE_bool(
    'basic_pitch_midi',
    False,
    help='Use basic pitch to obtain midi scores from the audio files',
    required=False)

flags.DEFINE_bool('beat_track',
                  False,
                  help='Use beat tracking to extract beats and downbats',
                  required=False)

flags.DEFINE_string(
    'waveform_augmentation',
    default="shift_stretch",
    help=
    "Perform data augmentation for the timbre input : [none, shift, stretch, shift_stretch]"
)

flags.DEFINE_integer('num_augments',
                     default=4,
                     help="Number of augmentations to perform")

flags.DEFINE_integer('num_multiprocesses',
                     default=4,
                     help="Number of processes for the data augmentation")
flags.DEFINE_multi_string('descriptors',
                          default=[],
                          help="Audio descriptors to compute")


def normalize_signal(x: np.ndarray,
                     max_gain_db: int = 30,
                     gain_margin: float = 0.9):
    peak = np.max(abs(x))
    if peak == 0:
        return x
    log_peak = 20 * np.log10(peak)
    log_gain = min(max_gain_db, -log_peak)
    gain = 10**(log_gain / 20)
    return gain_margin * x * gain


def get_midi(midi_data, chunk_number):
    length = FLAGS.num_signal / FLAGS.sample_rate
    tstart = chunk_number * FLAGS.num_signal / FLAGS.sample_rate
    tend = (chunk_number + 1) * FLAGS.num_signal / FLAGS.sample_rate
    out_notes = []
    for note in midi_data.instruments[0].notes:
        if note.end > tstart and note.start < tend:
            note.start = max(0, note.start - tstart)
            note.end = min(note.end - tstart, length)
            out_notes.append(note)

    if len(out_notes) == 0:
        return True, None
    midi_data.instruments[0].notes = out_notes
    midi_data.adjust_times([0, length], [0, length])
    return False, midi_data


def main(dummy):
    device = "cuda:" + str(
        FLAGS.gpu) if torch.cuda.is_available() and FLAGS.gpu >= 0 else "cpu"
    print("Using device : ", device)
    emb_model = None if FLAGS.emb_model_path is None else torch.jit.load(
        FLAGS.emb_model_path).to(device).eval()

    env = lmdb.open(
        FLAGS.output_path,
        map_size=FLAGS.db_size * 1024**3,
        map_async=True,
        writemap=True,
        readahead=False,
    )

    audio_files, midi_files, metadatas = get_parser(
        FLAGS.parser)(FLAGS.input_path, FLAGS.midi_path, FLAGS.ext,
                      FLAGS.exclude, FLAGS.include)

    chunks_buffer, metadatas_buffer = [], []
    midis = []
    cur_index = 0

    # Load BasicPitchPytorch
    if FLAGS.basic_pitch_midi:
        BP = BasicPitchPytorch(sr=FLAGS.sample_rate, device=device)
    else:
        BP = None

    if FLAGS.beat_track:
        beat_tracker = BeatTrack(sr=FLAGS.sample_rate, device=device)

    # Data augmentations
    if FLAGS.waveform_augmentation == "none":
        print("Using no augmentation")
        waveform_augmentation, waveform_pool = None, None

    else:
        if FLAGS.waveform_augmentation == "shift_stretch":
            waveform_augmentation = PSTS(ts_min=0.76,
                                         ts_max=1.49,
                                         pitch_min=-4,
                                         pitch_max=4,
                                         sr=FLAGS.sample_rate,
                                         chunk_size=FLAGS.num_signal // 4,
                                         random_silence=True)

        elif FLAGS.waveform_augmentation == "stretch":
            from after.dataset.transforms import TimeStretch

            waveform_augmentation = TimeStretch(sr=FLAGS.sample_rate,
                                                ts_min=0.7,
                                                ts_max=1.6,
                                                random_silence=True)

        elif FLAGS.waveform_augmentation == "shift":
            waveform_augmentation = PSTS(ts_min=1,
                                         ts_max=1,
                                         pitch_min=-6,
                                         pitch_max=6,
                                         sr=FLAGS.sample_rate,
                                         chunk_size=FLAGS.num_signal // 4)
        else:
            raise ValueError("Unknown waveform augmentation")
        waveform_pool = Pool(FLAGS.num_multiprocesses)

    # Audio descriptors

    if len(FLAGS.descriptors) > 0:
        waveform_descriptors = AudioDescriptors(sr=FLAGS.sample_rate,
                                                descriptors=FLAGS.descriptors,
                                                hop_length=512,
                                                n_fft=2048)
        if waveform_pool is None:
            waveform_pool = Pool(FLAGS.num_multiprocesses)

    # Processing loop
    for i, (file, midi_file, metadata) in enumerate(
            zip(tqdm(audio_files), midi_files, metadatas)):

        try:
            audio = librosa.load(file, sr=FLAGS.sample_rate, mono=True)[0]
        except:
            print("error loading file : ", file)
            continue

        audio = audio.squeeze()

        if audio.shape[-1] == 0:
            print("Empty file")
            continue

        if FLAGS.normalize:
            audio = normalize_signal(audio)

        # In case no midi_file is used, we can tile the audio file. Otherwise, we need to keep the alignement between midi data and audio.
        if midi_file is None:
            # Pad to a power of 2 if audio is longer than num_signal, tile if audio is too short
            if audio.shape[-1] > FLAGS.num_signal and audio.shape[
                    -1] % FLAGS.num_signal > FLAGS.num_signal // 2:
                audio = np.pad(
                    audio,
                    (0, FLAGS.num_signal - audio.shape[-1] % FLAGS.num_signal))
            elif audio.shape[-1] < FLAGS.num_signal:
                while audio.shape[-1] < FLAGS.num_signal:
                    audio = np.concatenate([audio, audio])

        audio = audio[:audio.shape[-1] // FLAGS.num_signal * FLAGS.num_signal]

        # MIDI DATA
        if (midi_file is None
                and BP is not None) and audio.shape[-1] > 4 * FLAGS.num_signal:
            if audio.shape[-1] > 4 * FLAGS.num_signal:
                chunk_size = 4 * FLAGS.num_signal
                audios = [
                    audio[..., i:i + chunk_size]
                    for i in range(0, audio.shape[-1], chunk_size)
                ]
                audios = [a for a in audios if a.shape[-1] >= FLAGS.num_signal]
        else:
            audios = [audio]

        for audio in audios:
            if midi_file is not None:
                midi_data = pretty_midi.PrettyMIDI(midi_file)

            elif midi_file is None and FLAGS.basic_pitch_midi:
                midi_data = BP(audio)

            else:
                midi_data = None

            # Reshape into chunks
            chunks = audio.reshape(-1, FLAGS.num_signal)
            chunk_index = 0

            for j, chunk in enumerate(chunks):
                # Chunk the midi
                if midi_data is not None:
                    silence_test, midi = get_midi(copy.deepcopy(midi_data),
                                                  chunk_number=chunk_index)
                else:
                    midi = None
                    silence_test = np.max(
                        abs(chunk)) < 0.05 if FLAGS.cut_silences else False

                # don't process buffer if empty slice
                if silence_test:
                    chunk_index += 1
                    continue

                midis.append(midi)
                chunks_buffer.append(chunk)
                metadatas_buffer.append(metadata)

                if len(chunks_buffer) == FLAGS.batch_size or (
                        j == len(chunks) - 1 and i == len(audio_files) - 1):

                    if emb_model is not None:
                        chunks_buffer_torch = torch.from_numpy(
                            np.stack(chunks_buffer)).to(device)

                        z = emb_model.encode(
                            chunks_buffer_torch.reshape(
                                -1, 1, FLAGS.num_signal))

                        # Data augmentations for the timbre
                        if waveform_augmentation is not None:
                            augments = {}
                            for i in range(FLAGS.num_augments):
                                augmented_buffers = waveform_pool.map(
                                    waveform_augmentation, chunks_buffer)
                                augmented_buffers_torch = [
                                    torch.from_numpy(a).reshape(1, 1,
                                                                -1).to(device)
                                    for a in augmented_buffers
                                ]
                                z_augmented = [
                                    emb_model.encode(
                                        a).squeeze().cpu().numpy()
                                    for a in augmented_buffers_torch
                                ]
                                augments["augment_" +
                                         FLAGS.waveform_augmentation + "_" +
                                         str(i)] = z_augmented
                        else:
                            augments = None

                        # Audio descriptors
                        features = {}
                        if len(FLAGS.descriptors) > 0:
                            descriptors_buffers = waveform_pool.map(
                                partial(waveform_descriptors,
                                        z_length=z.shape[-1]), chunks_buffer)
                            for k in descriptors_buffers[0]:
                                features[k] = [
                                    d[k] for d in descriptors_buffers
                                ]

                        # Beat tracking
                        if FLAGS.beat_track:
                            beat_data = [
                                beat_tracker(chunk, z_length=z.shape[-1])
                                for chunk in chunks_buffer
                            ]
                            features["beat_clock"] = [
                                b["beat_clock"] for b in beat_data
                            ]
                            features["downbeat_clock"] = [
                                b["downbeat_clock"] for b in beat_data
                            ]

                    else:
                        z = [None] * len(chunks_buffer)
                        augments = None
                        features = None

                    for k, (array, curz, midi, cur_metadata) in enumerate(
                            zip(chunks_buffer, z, midis, metadatas_buffer)):

                        ae = AudioExample()

                        if FLAGS.save_waveform:
                            assert array.shape[-1] == FLAGS.num_signal
                            array = (array * (2**15 - 1)).astype(np.int16)
                            ae.put_array("waveform", array, dtype=np.int16)

                        # EMBEDDING
                        if curz is not None:
                            ae.put_array("z",
                                         curz.cpu().numpy(),
                                         dtype=np.float32)

                        # METADATA
                        cur_metadata["chunk_index"] = chunk_index
                        ae.put_metadata(cur_metadata)

                        # MIDI DATA
                        if midi is not None:
                            ae.put_buffer(key="midi",
                                          b=pickle.dumps(midi),
                                          shape=None)

                        if augments is not None:
                            for key, augmented_buffers in augments.items():
                                ae.put_array(key,
                                             augmented_buffers[k],
                                             dtype=np.float32)

                        if features is not None:
                            for key, descr in features.items():
                                ae.put_array(key, descr[k], dtype=np.float32)

                        key = f"{cur_index:08d}"

                        with env.begin(write=True) as txn:
                            txn.put(key.encode(), bytes(ae))
                        cur_index += 1

                    chunks_buffer, midis, metadatas_buffer = [], [], []
                chunk_index += 1
    env.close()


if __name__ == '__main__':
    app.run(main)
