# RF-DETR Training & Inference on Modal

Train and run inference on [RF-DETR](https://github.com/roboflow/rf-detr) — Roboflow's real-time transformer-based object detector — on a serverless GPU with [Modal](https://modal.com). No local GPU required.

This repo is the companion code to the YouTube walkthrough: **[link coming soon]**.

<p align="center">
  <img src="assets/img1.png" width="49%" alt="RF-DETR detection example 1" />
  <img src="assets/img2.png" width="49%" alt="RF-DETR detection example 2" />
</p>

---

## What this demo shows

- Pulling a labeled object-detection dataset from Roboflow Universe
- Fine-tuning **RF-DETR Nano** on an L4 GPU in the cloud
- Persisting the dataset and trained checkpoints to a Modal Volume
- Running inference against the trained checkpoint on demand

The full pipeline lives in a single file — `rfdetr_training_inference.py` — so it's easy to read along with the video.

---

## Prerequisites

You'll need free accounts and CLI access for both:

| Service       | Why                                  | Sign up                              |
| ------------- | ------------------------------------ | ------------------------------------ |
| **Modal**     | Serverless GPU runtime               | https://modal.com/signup             |
| **Roboflow**  | Source of the labeled dataset        | https://app.roboflow.com/            |

Plus [`uv`](https://docs.astral.sh/uv/) for Python dependency management.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/rfdetr-training-inference.git
cd rfdetr-training-inference
uv sync
```

### 2. Authenticate with Modal

```bash
uv run modal setup
```

### 3. Add your Roboflow API key as a Modal secret

Grab your key from https://app.roboflow.com/settings/api, then:

```bash
uv run modal secret create roboflow-api-key ROBOFLOW_API_KEY=<your-key>
```

The function decorator looks this secret up by name — see `rfdetr_training_inference.py:48`.

---

## Run it

Kick off the full pipeline (dataset download → training):

```bash
uv run modal run rfdetr_training_inference.py
```

Modal will build the container image on first run (a few minutes), then stream training logs from the L4 GPU. Checkpoints land at `/root/data/checkpoints` inside the persisted volume.

---

## Configuration

All knobs live at the top of `rfdetr_training_inference.py`:

```python
EPOCHS = 10
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2

TRAIN_GPU = "L4"
TRAIN_CPU_COUNT = 4
INFERENCE_GPU = "L4"
```

Modal supports a wide range of GPUs — bump `TRAIN_GPU` to `"A100"` or `"H100"` for larger datasets or faster epochs. See the [Modal GPU reference](https://modal.com/docs/guide/gpu).

The dataset is selected inside `main()`:

```python
config = DatasetConfig(
    workspace_id="roboflow-100",
    project_id="vehicles-q0x2v",
    version=2,
    format="coco",
    target_class="vehicle",
)
```

Swap any Roboflow Universe project here. The Roboflow URL `https://universe.roboflow.com/<workspace>/<project>/dataset/<version>` maps directly onto these fields.

---

## How it works

```
┌────────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ download_dataset() │ ──► │     train()      │ ──► │    infer()       │
│  Roboflow API      │     │  RFDETRNano on   │     │ load checkpoint, │
│  → Modal Volume    │     │  L4 GPU          │     │ run prediction   │
└────────────────────┘     └──────────────────┘     └──────────────────┘
              shared `rfdetr_training_volume` (Modal Volume)
```

Each function is decorated with `@app.function(...)` and runs in its own container. The Modal Volume mounted at `/root/data` keeps the dataset and checkpoints around between runs so you don't re-download or re-train every time.

Note: heavyweight imports (`rfdetr`, `roboflow`) are deferred to inside the function bodies because they're only installed in the remote container image, not locally.

---

## FiftyOne dataset exploration

A separate local script — `fiftyone_workflow.py` — lets you explore the vegetables-detector dataset visually before (or after) training. It runs entirely on your machine; Modal is only needed for GPU training.

### What it computes

| Step | What happens | FiftyOne key / field |
|------|--------------|----------------------|
| Dataset import | All three COCO splits (train / valid / test) merged into one persistent dataset, each sample tagged with its split | `ground_truth` label field, `train` / `valid` / `test` tags |
| CLIP embeddings + UMAP | 2-D projection of every image's CLIP embedding — lets you spot clusters, outliers, and near-duplicates in the App's **Embeddings** panel | `clip_umap` brain key |
| Similarity index | Nearest-neighbour index over CLIP embeddings — enables the **Find similar** button in the App | `clip_similarity` brain key |
| Image quality | Per-sample brightness, blurriness, exposure, entropy, aspect ratio — surfaced as numeric fields you can sort/filter | plugin fields **or** `quality_brightness`, `quality_blurriness`, `quality_aspect_ratio` fallback fields |

### Install steps

```bash
# 1. Sync all local dependencies (fiftyone, umap-learn, open-clip-torch, Pillow, numpy)
uv sync

# 2. (Recommended) Install the image-quality plugin for richer quality metrics
uv run fiftyone plugins download https://github.com/jacobmarks/image-quality-issues
```

### Set your Roboflow API key

```bash
export ROBOFLOW_API_KEY="<your-roboflow-api-key>"
```

### Run the workflow

```bash
uv run python fiftyone_workflow.py
```

The script will:
1. Download the dataset to `./data/vegetables_detector_annotation-sud2e/1/` (skipped on re-runs if already present).
2. Build a persistent FiftyOne dataset named `vegetables_detector` (or load it if it already exists).
3. Compute CLIP embeddings → UMAP projection → similarity index.
4. Run image-quality analysis (plugin if installed, PIL/numpy fallback otherwise).
5. Open the FiftyOne App at **http://localhost:5151** in your browser.

Press **Ctrl-C** in the terminal when you're done.

### Exploring in the App

- **Embeddings panel** → select brain key `clip_umap` to see the 2-D cluster view.
- **Similarity** → click any image then "Find similar" to surface near-duplicates.
- **Filters sidebar** → sort/filter by `quality_brightness`, `quality_blurriness`, etc. to surface low-quality images.
- **Tags** → filter by `train`, `valid`, or `test` to inspect each split independently.

### Re-opening the dataset later

The FiftyOne dataset is stored in your local FiftyOne DB (`~/.fiftyone/`). To reload it any time without re-running the full workflow:

```python
import fiftyone as fo
dataset = fo.load_dataset("vegetables_detector")
fo.launch_app(dataset)
```

---

## File layout

```
.
├── rfdetr_training_inference.py   # Modal training + inference pipeline
├── fiftyone_workflow.py           # Local FiftyOne exploration script
├── pyproject.toml                 # Dependencies (uv-managed)
├── uv.lock
└── README.md
```

---

## Costs

Modal grants new accounts $30/month of free compute, which is plenty for this demo. An L4 training run on the example dataset (~10 epochs) finishes in a few minutes and costs cents.

---

## Troubleshooting

- **`Secret 'roboflow-api-key' not found`** — re-run the secret-create command in Setup step 3.
- **Build is slow on first run** — Modal caches the image after the first build; subsequent runs start in seconds.
- **OOM during training** — drop `BATCH_SIZE` or bump to a larger GPU.

---

## License

MIT
