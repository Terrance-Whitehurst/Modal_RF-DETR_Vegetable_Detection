from dataclasses import dataclass
from pathlib import Path
import os

import modal

MINUTES = 60

EPOCHS = 10
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2

TRAIN_GPU = "L4"
TRAIN_CPU_COUNT = 4
INFERENCE_GPU = "L4"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .uv_pip_install("rfdetr", "roboflow", gpu=TRAIN_GPU)
)

volume = modal.Volume.from_name("rfdetr_training_volume", create_if_missing=True)
volume_path = Path("/root") / "data"

app = modal.App(
    "rfdetr_training_and_inference",
    image=image,
    volumes={volume_path: volume},
)


@dataclass
class DatasetConfig:
    """Information required to download a dataset from Roboflow."""

    workspace_id: str
    project_id: str
    version: int
    format: str
    target_class: str

    @property
    def id(self) -> str:
        return f"{self.workspace_id}/{self.project_id}/{self.version}"


@app.function(
    secrets=[modal.Secret.from_name("roboflow-api-key", required_keys=["ROBOFLOW_API_KEY"])],
)
def download_dataset(config: DatasetConfig) -> str:
    from roboflow import Roboflow

    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    project = (
        rf.workspace(config.workspace_id)
        .project(config.project_id)
        .version(config.version)
    )
    dataset_dir = volume_path / "dataset" / config.id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    project.download(config.format, location=str(dataset_dir))
    volume.commit()
    return str(dataset_dir) 


@app.function(
    gpu=TRAIN_GPU,
    cpu=TRAIN_CPU_COUNT, 
    timeout=60 * MINUTES,
)
def train(dataset_dir: str) -> str:
    from rfdetr import RFDETRNano

    output_dir = volume_path / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = RFDETRNano()
    model.train(
        dataset_dir=dataset_dir,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        output_dir=str(output_dir),
    )
    volume.commit()
    return str(output_dir)


@app.function(gpu=INFERENCE_GPU, timeout=10 * MINUTES)
def infer(checkpoint_dir: str, image_path: str, threshold: float = 0.5):
    from rfdetr import RFDETRNano

    weights = Path(checkpoint_dir) / "checkpoint_best_total.pth"
    model = RFDETRNano(pretrain_weights=str(weights))
    detections = model.predict(image_path, threshold=threshold)
    return detections


@app.local_entrypoint()
def main():
    config = DatasetConfig(
        workspace_id="roboflow-100",
        project_id="vehicles-q0x2v",
        version=2,
        format="coco",
        target_class="vehicle",
    )
    dataset_dir = download_dataset.remote(config)
    checkpoint_dir = train.remote(dataset_dir)
    print(f"Training complete. Checkpoints at: {checkpoint_dir}")
