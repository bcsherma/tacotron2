# Tacotron 2 + Weights & Biases

PyTorch implementation of [Natural TTS Synthesis By Conditioning
Wavenet On Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf).
This fork has been instrumented with Weights & Biases to enable experiment
tracking, prediction logging, dataset and model versioning, and hyperparameter 
optimziation.

This implementation includes uses the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/).

## Pre-requisites
1. NVIDIA GPU + CUDA cuDNN

## Running
1. Run `pip install -r requirements.txt`
2. Run `wandb init` to configure your working directory to log to Weights & Biases.
3. Run `python register-data.py` to create a reference Artifact pointing to the LJSpeech dataset.
4. Run `python split-data.py` to create a versioned train/validation split of the data.
5. Run `python register-model ...` to log pre-trained tacotron and waveglow models as Artifacts to Weights & Biases.
6. Run `python train.py <dataset-artifact>` to warm-start train tacotron2 on the dataset you created.
7. Run `python inference.py <tacotron-artifact> <waveglow-artifact> <text>` to run inference on a text file containing newline delimited sentences. The inference results will be logged to Weights & Biases as a `wandb.Table`

## Related repos
[WaveGlow](https://github.com/NVIDIA/WaveGlow) Faster than real time Flow-based
Generative Network for Speech Synthesis

[nv-wavenet](https://github.com/NVIDIA/nv-wavenet/) Faster than real time
WaveNet.

## Acknowledgements (Copied)
This implementation uses code from the following repos: [Keith
Ito](https://github.com/keithito/tacotron/), [Prem
Seetharaman](https://github.com/pseeth/pytorch-stft) as described in our code.

We are inspired by [Ryuchi Yamamoto's](https://github.com/r9y9/tacotron_pytorch)
Tacotron PyTorch implementation.

We are thankful to the Tacotron 2 paper authors, specially Jonathan Shen, Yuxuan
Wang and Zongheng Yang.


[WaveGlow]: https://drive.google.com/open?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF
[Tacotron 2]: https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view?usp=sharing
[pytorch 1.0]: https://github.com/pytorch/pytorch#installation
[website]: https://nv-adlr.github.io/WaveGlow
[ignored]: https://github.com/NVIDIA/tacotron2/blob/master/hparams.py#L22
[Apex]: https://github.com/nvidia/apex
[AMP]: https://github.com/NVIDIA/apex/tree/master/apex/amp
