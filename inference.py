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


def main(tacotron_artifact, waveglow_artifact, sentence_file):

    with open(sentence_file) as infile:
        sentences = [
            sentence.strip() for sentence in infile.readlines() if sentence.strip()
        ]

    # Initialize inference table
    inference_table = wandb.Table(columns=["sentence", "audio"])

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

    # Tokenize sentences
    sequences = [
        torch.autograd.Variable(
            torch.from_numpy(
                np.array(text_to_sequence(text, ["english_cleaners"]))[None, :]
            )
        )
        .cuda()
        .long()
        for text in sentences
    ]

    # Run inference for each sequence
    for sequence, sentence in zip(sequences, sentences):
        _, mel_outputs_postnet, _, _ = model.inference(sequence)
        with torch.no_grad():
            audio = waveglow.infer(mel_outputs_postnet)
            audio = audio.cpu().numpy().astype(np.float32)
            inference_table.add_data(
                sentence, wandb.Audio(audio[0], sample_rate=HPARAMS["sampling_rate"])
            )

    wandb.log({"inference": inference_table})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tacotron", help="tacotron model artifact to load", type=str)
    parser.add_argument("waveglow", help="waveglow model artifact to load", type=str)
    parser.add_argument(
        "sentences",
        help="text file containing sentences to be converted to audio",
        type=str,
    )
    args = parser.parse_args()
    main(args.tacotron, args.waveglow, args.sentences)
