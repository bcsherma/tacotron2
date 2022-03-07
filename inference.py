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

# Needed to avoid warnings
torch.nn.Module.dump_patches = True


with open("hparams.yaml") as yamlfile:
    HPARAMS = yaml.safe_load(yamlfile)


def main(tacotron_artifact, waveglow_artifact, sentence_file, table_artifact):

    # Initialize inference table
    inference_table = wandb.Table(
        columns=["predicted_caption" if table_artifact else "sentence", "audio"]
    )

    # Start wandb tracking and get artifacts
    # run = wandb.init(job_type="inference", entity="faux", project="image-to-speech")
    wandb.init(job_type="inference")
    if sentence_file:
        wandb.config.update({"source": sentence_file})
    taco_artifact = wandb.use_artifact(tacotron_artifact)
    wave_artifact = wandb.use_artifact(waveglow_artifact)

    # Load sentences for inference
    if sentence_file:
        with open(sentence_file) as infile:
            sentences = [
                sentence.strip() for sentence in infile.readlines() if sentence.strip()
            ]
    
    elif table_artifact:
        table_artifact = wandb.use_artifact(table_artifact)
        table = table_artifact.get("eval_table")
        sentences = table.get_column("predicted_caption")
        sentences = [(s, s.replace("<end>", "")) for s in sentences][:20]

    # Load tacotron model
    checkpoint_path = taco_artifact.get_path("model.pt").download()
    model = load_model(HPARAMS)
    model.load_state_dict(torch.load(checkpoint_path))
    model.cuda().eval()

    # Load waveglow model
    waveglow_path = wave_artifact.get_path("pretrained-model.pt").download()
    waveglow = torch.load(waveglow_path)["model"]
    waveglow.cuda().eval()
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
            audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
            audio = audio.cpu().numpy().astype(np.float32)
            inference_table.add_data(
                sentence[0] if table_artifact else sentence,
                wandb.Audio(audio[0], sample_rate=HPARAMS["sampling_rate"]),
            )

    if table_artifact:
        inference_table = wandb.JoinedTable(table, inference_table, "predicted_caption")

    wandb.log({"eval_table": inference_table})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tacotron", help="tacotron model artifact to load", type=str)
    parser.add_argument("waveglow", help="waveglow model artifact to load", type=str)
    parser.add_argument(
        "-s",
        "--sentences",
        help="text file containing sentences to be converted to audio",
        type=str,
        metavar="sentences",
    )
    parser.add_argument(
        "-t",
        "--table",
        help="Artifact containing table with sentences to predict on.",
    )
    args = parser.parse_args()
    if bool(args.sentences) == bool(args.table):
        print("Only one of --sentences and --table may be specified.")
        exit(1)
    main(args.tacotron, args.waveglow, args.sentences, args.table)
