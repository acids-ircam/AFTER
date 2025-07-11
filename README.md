![After Logo](/docs/after_nobackground.png)

# AFTER: Audio Features Transfer and Exploration in Real-time

__AFTER__ is a diffusion-based generative model that creates new audio by blending two sources: one audio stream to set the style or timbre, and another input (either audio or MIDI) to shape the structure over time.

This repository is a real-time implementation of the research paper _Combining audio control and style transfer using latent diffusion_ ([read it here](https://arxiv.org/abs/2408.00196)) by Nils Demerlé, P. Esling, G. Doras, and D. Genova. Some transfer examples can be found on the [project webpage](https://nilsdem.github.io/control-transfer-diffusion/). This real-time version integrates with MaxMSP and Ableton Live through [_nn_tilde_](https://github.com/acids-ircam/nn_tilde), an external that embeds PyTorch models into MaxMSP.

You can find pretrained models and max patches for realtime inference in the last section of this page.

### Installation

``` bash
git clone https://github.com/acids-ircam/AFTER.git
cd AFTER/
pip install -e .
```

If you want to use the model in MaxMSP or PureData for real-time generation, please refer to the [_nn_tilde_](https://github.com/acids-ircam/nn_tilde) external documentation and follow the installation steps.

## Model Training

Training AFTER involves 3 separate steps, _autoencoder training_, _model training_ and _model export_.

### Neural audio codec

If you already have a streamable audio codec such as a pretrained [RAVE](https://github.com/acids-ircam/RAVE) model, you can directly skip to the next section. Also, we provide four audio codecs already trained on different datasets [here](https://nubo.ircam.fr/index.php/s/8NFD5gWwbkT4G5P).

Before training the autoencoder, you need to preprocess your audio files into an lmdb database :

```bash
after prepare_dataset --input_path /audio/folder --output_path /dataset/path_audio --save_waveform True 
```

Then, you can start the model training 

```bash
after train_autoencoder --name AE_model_name --db_path /dataset/path_audio  --config baseAE --gpu 0
```

where `db_path` refers to the prepared dataset location. The tensorboard logs and checkpoints are saved by default to `./autoencoder_runs/`.

After training, the model has to be exported to a torchscript file using

```bash
after export_autoencoder  --model_path autoencoder_runs/AE_model_name 
```
This will save two .ts files in the run folder, one for streaming and one for offline inference (export_stream.ts and export.ts respectively). By default the last training step checkpoint is used for export.

### AFTER training
First, you need to prepare your dataset before training. Since our diffusion model works in the latent space of the autoencoder, we pre-compute the latent embeddings to speed up training : 

```bash
after prepare_dataset --input_path /audio/folder --output_path /dataset/path_latent_codes --emb_model_path AE_model_run_path/export.ts
```

- `num_signal` flag sets the duration of the audio chunks for training in number of samples (must be a power of 2). (default: 524288 ~ 11 seconds)
- `sample_rate` flag sets the resampling rate.  (default: 44100)
- `gpu` device to use for computing the embeddings. Use -1 for cpu (default: 0)

To train a midi-to-audio AFTER model you need to either use the flag `--basic_pitch_midi` to transcript the midi from the audio files or define your own file parsing function in `./after/dataset/parsers.py`.

If you plan to have more advanced use of the models, please refer to the help function for all the arguments.

Then, a training is started with 

```bash
after train  --name diff_model_name --db_path /dataset/path_latent_codes --emb_model_path AE_model_run_path/export.ts --config CONFIG_NAME
```

Different configurations are available in `diffusion/configs` and can be combined : 


<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Config</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><strong>Model</strong></td>
      <td><strong>base</strong></td>
      <td>Default audio-to-audio timbre and structure separation model.</td>
    </tr>
    <tr>
      <td><strong>midi</strong></td>
      <td>Uses MIDI as input for the structure encoder</td>
    </tr>
    <tr>
      <td rowspan="2"><strong>Additional</strong></td>
      <td><strong>tiny</strong></td>
      <td>Reduces the model's capacity for faster inference. Useful for testing and low-resource environments.</td>
    </tr>
    <tr>
      <td><strong>cycle</strong></td>
      <td>Experimental: adds a cycle consistency phase during training, which can improve timbre and structure disentanglement.</td>
    </tr>
  </tbody>
</table>


The tensorboard logs and checkpoints are saved to  `/diffusion/runs/model_name`, and you can experiment with you trained model using the notebooks `notebooks/audio_to_audio_demo.ipynb` and `notebooks/midi_to_audio_demo.ipynb`.

### Export

Once the training is complete, you can export the model to an [_nn_tilde_](https://github.com/acids-ircam/nn_tilde) torchscript file for inference in MaxMSP, PureData or Ableton Live. 

For an audio-to-audio model :
```bash
after export --model_path diff_model_name --emb_model_path AE_model_run_path/export_stream.ts
```

For a MIDI-to-audio model :

```bash
after export_midi --model_path diff_model_name --emb_model_path AE_model_run_path/export_stream.ts 
```

Make sure to use the streaming version of the exported autoencoder (denoted by _stream.ts).

## Inference

You can experiment with inference in MaxMSP using the patches in `./patchs` and the pretrained models available [here](https://nubo.ircam.fr/index.php/s/8NFD5gWwbkT4G5P).

We also provide two Max4Live devices to use your models in Ableton Live. By default the export scripts trains a small network to remap the timbre latent space to a 2D map that can be used for latent exploration in our Max4Live device (see below). If you use multiple datasets, each dataset will correspond to one color on the latent map. The 2D map is used for coarse latent control, which you can refine by directly changing the latent dimensions. Make sure to download the .ts file along with .png latent map created with the export script.

<p align="center">
  <img src="docs/after_midi.png" />
  <br/>
  <img src="docs/after_audio.png" />
</p>


<!-- ### MIDI-to-Audio 

Our MIDI-to-audio model is a 4-voice polyphonic synthesizer that produces audio for pitch and velocity, as well as a timbre target in two modes:
- **Audio-based**: Using the `forward` method, AFTER extracts timbre from an audio stream (with a 3 seconds receptive field). We’ve included audio samples from the training set in the repository.
- **Manual exploration**: The `forward_manual` method lets you explore timbre with 8 sliders, which set a position in a learned 8-dimensional timbre space.

The guidance parameter sets the conditioning strength on the MIDI input, and diffusion steps can be adjusted to improve generation quality (at a higher CPU cost).

Download our instrumental model trained on the [SLAKH](http://www.slakh.com/) dataset [here](https://nubo.ircam.fr/index.php/s/tHMmFmkF6kgn7ND/download).

Audio Timbre Target           |  Manual Timbre Control
:-------------------------:|:-------------------------:
<img src="docs/midi_to_audio.png"   height="500"/>| <img src="docs/midi_to_audio_manual.png"  height="500"/>



### Audio-to-Audio 

In audio-to-audio mode, AFTER extracts the time-varying features from one audio stream and applies them to the timbre of a second audio source. The guidance parameter controls the conditioning strength on the structure input, and the diffusion steps improve generation quality with more CPU load.

Download our instrumental model trained on the [SLAKH](http://www.slakh.com/) dataset [here](https://nubo.ircam.fr/index.php/s/NCHZ5Q9aMsFxmyp/download).

<img src="docs/audio_to_audio.png"  height="500"/> -->

## Artistic Applications

AFTER has been applied in several projects:
- [_The Call_](https://www.serpentinegalleries.org/whats-on/holly-herndon-mat-dryhurst-the-call/) by Holly Herndon and Mat Dryhurst, an interactive sound installation with singing voice transfer, at Serpentine Gallery in London until February 2, 2025.
- A live performance by French electronic artist Canblaster for Forum Studio Session at IRCAM. The full concert is available on [YouTube](https://www.youtube.com/watch?v=0E9nNyz4pv4).
- [Nature Manifesto](https://www.centrepompidou.fr/fr/programme/agenda/evenement/dkTTgJv), an immersive sound installation by Björk and Robin Meier, at Centre Pompidou in Paris from November 20 to December 9, 2024.

We look forward to seeing new projects and creative uses of AFTER.
