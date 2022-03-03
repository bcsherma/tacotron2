import argparse
import os
import tarfile

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

import wandb
from data_utils import TextMelCollate, TextMelLoader
from loss_function import Tacotron2Loss
from model import Tacotron2
from plotting_utils import plot_spectrogram_to_numpy


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams["training_files"], hparams)
    valset = TextMelLoader(hparams["validation_files"], hparams)
    collate_fn = TextMelCollate(hparams["n_frames_per_step"])

    train_sampler = None
    shuffle = True

    train_loader = DataLoader(
        trainset,
        num_workers=1,
        shuffle=shuffle,
        sampler=train_sampler,
        batch_size=hparams["batch_size"],
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return train_loader, valset, collate_fn


def load_model(hparams):
    return Tacotron2(hparams).cuda()


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model_dict = checkpoint_dict["state_dict"]
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items() if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint_dict["state_dict"])
    optimizer.load_state_dict(checkpoint_dict["optimizer"])
    learning_rate = checkpoint_dict["learning_rate"]
    iteration = checkpoint_dict["iteration"]
    print("Loaded checkpoint '{}' from iteration {}".format(checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def validate(
    model,
    criterion,
    valset,
    iteration,
    batch_size,
    collate_fn,
):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_loader = DataLoader(
            valset,
            sampler=None,
            num_workers=1,
            shuffle=False,
            batch_size=batch_size,
            pin_memory=False,
            collate_fn=collate_fn,
        )
        table = wandb.Table(
            columns=["step", "sentence", "audio", "ground truth", "prediction"]
        )
        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
            inputs = valset.audiopaths_and_text[batch_size * i : batch_size * (i + 1)]
            _, mel_outputs, _, _ = y_pred
            mel_targets, _ = y
            for (audio, sentence), y, y_pred in zip(inputs, mel_targets, mel_outputs):
                table.add_data(
                    iteration,
                    sentence,
                    wandb.Audio(audio),
                    wandb.Image(plot_spectrogram_to_numpy(y.data.cpu().numpy())),
                    wandb.Image(plot_spectrogram_to_numpy(y_pred.data.cpu().numpy())),
                )
        val_loss = val_loss / (i + 1)

    model.train()
    print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
    wandb.log({"predictions": table, "validation/loss": val_loss})


def prepare_dataset(dataset):

    try:
        os.mkdir("./filelists/")
    except OSError:
        pass

    data_art = wandb.use_artifact(dataset)

    meta = pd.read_csv(
        data_art.get_path("transcriptions").download(),
        sep="|",
        names=["file", "sentence"],
        index_col=0,
    )

    for split in ["train", "validation"]:
        path = data_art.get_path(f"{split}.tar.bz2").download()
        filelist = []
        with tarfile.open(path, "r:bz2") as tarball:
            tarball.extractall(f"data/{split}/")
            for file in tarball.getnames():
                name = file.split(".")[0]
                filelist.append([f"data/{split}/{file}", meta.loc[name, "sentence"]])
        filelist = pd.DataFrame(filelist)
        filelist.to_csv(f"filelists/{split}.txt", sep="|", header=False, index=False)


def train(
    checkpoint_path,
    hparams,
    dataset,
):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    checkpoint_path (string): checkpoint path
    hparams (object): comma separated list of "name=value" pairs.
    dataset (string): data artifact to be loaded for training and validation
    """

    wandb.init(job_type="train", config=hparams)

    prepare_dataset(dataset)

    torch.manual_seed(hparams["seed"])
    torch.cuda.manual_seed(hparams["seed"])

    model = load_model(hparams)
    learning_rate = hparams["learning_rate"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=hparams["weight_decay"]
    )

    criterion = Tacotron2Loss()

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        model_artifact = wandb.use_artifact(checkpoint_path)
        path = model_artifact.get_path("pretrained-model.pt").download()
        model = warm_start_model(path, model, hparams["ignore_layers"])
    model.train()
    is_overflow = False
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams["epochs"]):
        print("Epoch: {}".format(epoch))
        for _, batch in enumerate(train_loader):
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch)
            y_pred = model(x)

            loss = criterion(y_pred, y)
            reduced_loss = loss.item()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), hparams["grad_clip_thresh"]
            )

            optimizer.step()

            wandb.log(
                {
                    "train/loss": reduced_loss,
                    "train/grad_norm": grad_norm,
                }
            )

            if (
                not is_overflow
                and (iteration % hparams["iters_per_checkpoint"] == 0)
                and iteration > 0
            ):
                validate(
                    model,
                    criterion,
                    valset,
                    iteration,
                    hparams["batch_size"],
                    collate_fn,
                )

            iteration += 1

    # Save final model as an Artifact
    torch.save(model.state_dict(), "model.pt")
    model_artifact = wandb.Artifact("tacotron2", type="model")
    model_artifact.add_file("model.pt")
    wandb.log_artifact(model_artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="split-ljs:latest",
        help="<artifact:version> formatted path to dataset artifact",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_artifact",
        type=str,
        default="tacotron-pretrained:latest",
        required=False,
        help="checkpoint artifact name",
    )
    parser.add_argument("--learning_rate", default=None, type=float)
    parser.add_argument("--weight_decay", default=None, type=float)

    args = parser.parse_args()

    with open("hparams.yaml") as yamlfile:
        hparams = yaml.safe_load(yamlfile)
    
    if args.learning_rate:
        hparams["learning_rate"] = args.learning_rate
    if args.weight_decay:
        hparams["weight_decay"] = args.weight_decay

    torch.backends.cudnn.enabled = hparams["cudnn_enabled"]
    torch.backends.cudnn.benchmark = hparams["cudnn_benchmark"]   

    print("Dynamic Loss Scaling:", hparams["dynamic_loss_scaling"])
    print("cuDNN Enabled:", hparams["cudnn_enabled"])
    print("cuDNN Benchmark:", hparams["cudnn_benchmark"])

    train(
        args.checkpoint_artifact,
        hparams,
        args.dataset,
    )
