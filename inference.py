import argparse
import sys

import torch
import yaml
import numpy as np

import wandb
from train import load_model
from text import text_to_sequence

# Needed to unpickle waveglow model
sys.path.append("./waveglow/")

with open("hparams.yaml") as yamlfile:
    HPARAMS = yaml.safe_load(yamlfile)


def main(tacotron_artifact, waveglow_artifact):

    # Start wandb tracking and get artifacts
    wandb.init(job_type="inference")
    taco_artifact = wandb.use_artifact(tacotron_artifact)
    wave_artifact = wandb.use_artifact(waveglow_artifact)

    # Load tacotron model
    checkpoint_path = taco_artifact.get_path("model.pt").download()
    model = load_model(HPARAMS)
    model.load_state_dict(torch.load(checkpoint_path))
    _ = model.cuda().eval().half()

    # Load waveglow model
    waveglow_path = wave_artifact.get_path("pretrained-model.pt").download()
    waveglow = torch.load(waveglow_path)["model"]
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()

    # Get and prepare text for inference.
    text = "Weights and Biases is a great product"
    sequence = np.array(text_to_sequence(text, ["english_cleaners"]))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

    # Run inference
    _, mel_outputs_postnet, _, _ = model.inference(sequence)
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet)
        audio = audio.cpu().numpy().astype(np.float32)
    wandb.log({"prediction": wandb.Audio(audio[0], sample_rate=HPARAMS["sampling_rate"])})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tacotron", help="tacotron model artifact to load", type=str)
    parser.add_argument("waveglow", help="waveglow model artifact to load", type=str)
    args = parser.parse_args()
    main(args.tacotron, args.waveglow)
