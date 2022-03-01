import wandb

DATA_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"


def register_data(url=DATA_URL):
    run = wandb.init(job_type="register-data")
    data_artifact = wandb.Artifact("ljs-tarball", type="raw data")
    data_artifact.add_reference(url, name="tarball")
    run.log_artifact(data_artifact)


if __name__ == "__main__":
    register_data()
