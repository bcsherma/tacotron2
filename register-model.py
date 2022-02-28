import argparse

import wandb


def register_model(path):
    run = wandb.init(job_type="register-model")
    data_artifact = wandb.Artifact("tacotron-pretrained", type="pretrained model")
    data_artifact.add_file(path, name="pretrained-model.pt")
    run.log_artifact(data_artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="path to model on disk", type=str)
    args = parser.parse_args()
    register_model(args.model)
