import os
import random
import tarfile

import pandas as pd
import wandb


def split_dataset(source_artifact, n_train, n_validation):
    """Split raw data artifact into train and validation sets.

    Args:
        source_artifact: <artifact:version> formatted raw data artifact path.
        n_train: Number of examples to include in the train set.
        n_validation: Number of examples to include in the validation set.
    """

    # Initialize wandb Run of type split-data
    run = wandb.init(job_type="split-data")

    # Download the raw data
    source = run.use_artifact(source_artifact)
    tarball_path = source.get_path("tarball").download()

    # Extract raw data
    tarball = tarfile.open(tarball_path, "r:bz2")
    tarball.extractall()

    # Construct new artifact
    split_dataset = wandb.Artifact(
        "split-ljs",
        type="split data",
        metadata={
            "train-examples": n_train,
            "val-examples": n_validation,
        },
    )

    # Add transcription data to artifact
    split_dataset.add_file("LJSpeech-1.1/metadata.csv", name="transcriptions")

    meta = pd.read_csv(
        "LJSpeech-1.1/metadata.csv",
        sep="|",
        names=["file", "sentence"],
        index_col=0,
    )

    # Get a list of all wav files and randomize the order
    all_files = os.listdir("LJSpeech-1.1/wavs")
    assert n_train + n_validation <= len(all_files)
    random.shuffle(all_files)
    idx = 0

    # Construct a tarball for each split of the data
    for size, split in (
        (n_train, "train"),
        (n_validation, "validation"),
    ):
        with tarfile.open(f"{split}.tar.bz2", "w:bz2") as tarball:
            jdx = 0
            while jdx < size:
                if not meta.loc[all_files[idx].split(".")[0], "sentence"].strip():
                    idx += 1
                    continue
                tarball.add(
                    f"LJSpeech-1.1/wavs/{all_files[idx]}", arcname=f"{all_files[idx]}"
                )
                idx += 1
                jdx += 1
        split_dataset.add_file(f"{split}.tar.bz2")

    # Log final artifact
    run.log_artifact(split_dataset)


if __name__ == "__main__":
    split_dataset("ljs-tarball:latest", 1024, 128)
