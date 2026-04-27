"""
FiftyOne Dataset Exploration for Vegetables Detector
=====================================================

Runs **locally** (not on Modal). Builds a persistent FiftyOne dataset from the
Roboflow vegetables-detector project, computes CLIP embeddings + UMAP
visualisation, flags image-quality issues, then launches the FiftyOne App so
you can explore everything interactively in your browser.

Usage
-----
    # 1. Ensure ROBOFLOW_API_KEY is exported in your shell
    export ROBOFLOW_API_KEY="<your-key>"

    # 2. (First time) install the image-quality plugin
    uv run fiftyone plugins download https://github.com/jacobmarks/image-quality-issues

    # 3. Run the workflow
    uv run python fiftyone_workflow.py

The FiftyOne App will open automatically at http://localhost:5151.
Press Ctrl-C in the terminal when you're done.

Dataset persistence
-------------------
The FiftyOne dataset is stored in the default FiftyOne database
(~/.fiftyone/mongod).  Re-running the script loads the existing dataset
instead of re-building it from scratch, so embeddings/quality fields
computed in a previous run are preserved.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROBOFLOW_WORKSPACE = "flovision"
ROBOFLOW_PROJECT = "vegetables_detector_annotation-sud2e"
ROBOFLOW_VERSION = 1
ROBOFLOW_FORMAT = "coco"

DATASET_NAME = "vegetables_detector"
LOCAL_DATA_ROOT = Path("./data")
DATASET_DIR = LOCAL_DATA_ROOT / ROBOFLOW_PROJECT / str(ROBOFLOW_VERSION)

SPLITS = ["train", "valid", "test"]

CLIP_MODEL = "clip-vit-base32-torch"
VISUALIZATION_KEY = "clip_umap"
SIMILARITY_KEY = "clip_similarity"


# ---------------------------------------------------------------------------
# 1. Download dataset from Roboflow
# ---------------------------------------------------------------------------


def download_dataset() -> Path:
    """Download the vegetables-detector dataset (COCO format) from Roboflow.

    Returns the path to the downloaded dataset root directory.
    Reads ROBOFLOW_API_KEY from the environment — never hardcoded.
    """
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: ROBOFLOW_API_KEY environment variable is not set.\n"
            "Export it with:\n"
            "  export ROBOFLOW_API_KEY='<your-roboflow-api-key>'\n"
            "You can find your key at https://app.roboflow.com/settings/api"
        )

    # Skip download if data already looks present (any split dir exists)
    if any((DATASET_DIR / split).exists() for split in SPLITS):
        print(f"[download] Dataset already present at {DATASET_DIR} — skipping download.")
        return DATASET_DIR

    print(f"[download] Downloading {ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT} v{ROBOFLOW_VERSION} ...")
    try:
        from roboflow import Roboflow  # deferred: not installed in all envs
    except ImportError:
        sys.exit(
            "ERROR: 'roboflow' package is not installed.\n"
            "Run:  uv add roboflow  or  pip install roboflow"
        )

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    project = (
        rf.workspace(ROBOFLOW_WORKSPACE)
        .project(ROBOFLOW_PROJECT)
        .version(ROBOFLOW_VERSION)
    )
    project.download(ROBOFLOW_FORMAT, location=str(DATASET_DIR))
    print(f"[download] Done. Dataset at {DATASET_DIR}")
    return DATASET_DIR


# ---------------------------------------------------------------------------
# 2. Build (or load) a persistent FiftyOne dataset
# ---------------------------------------------------------------------------


def build_fiftyone_dataset(dataset_dir: Path) -> "fo.Dataset":
    """Import all COCO splits into a single persistent FiftyOne dataset.

    If the dataset already exists in the FiftyOne DB it is loaded directly,
    preserving any previously computed fields (embeddings, quality metrics).

    Roboflow COCO layout (per split)::

        {split}/
            _annotations.coco.json   ← labels file
            *.jpg / *.png            ← images (same directory)

    FiftyOne's COCODetectionDataset expects:
        data_path   = directory that contains the images
        labels_path = path to the COCO JSON file
    """
    import fiftyone as fo  # deferred heavy import

    if fo.dataset_exists(DATASET_NAME):
        print(f"[dataset] Loading existing FiftyOne dataset '{DATASET_NAME}' ...")
        dataset = fo.load_dataset(DATASET_NAME)
        print(f"[dataset] Loaded {len(dataset)} samples.")
        return dataset

    print(f"[dataset] Creating FiftyOne dataset '{DATASET_NAME}' ...")
    dataset = fo.Dataset(DATASET_NAME)
    dataset.persistent = True

    for split in SPLITS:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            print(f"[dataset]   Split '{split}' not found at {split_dir} — skipping.")
            continue

        # Roboflow names the annotation file _annotations.coco.json
        labels_path = split_dir / "_annotations.coco.json"
        if not labels_path.exists():
            print(f"[dataset]   No annotation file found for split '{split}' — skipping.")
            continue

        print(f"[dataset]   Importing split '{split}' from {split_dir} ...")
        split_view = fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=str(split_dir),
            labels_path=str(labels_path),
            label_field="ground_truth",
            include_id=True,
        )

        # Tag every sample with its split name so you can filter in the App
        split_view.tag_samples(split)

        dataset.merge_samples(split_view)
        print(f"[dataset]   Added {len(split_view)} samples for split '{split}'.")

    dataset.save()
    print(f"[dataset] Dataset '{DATASET_NAME}' created with {len(dataset)} samples.")
    return dataset


# ---------------------------------------------------------------------------
# 3. Compute CLIP embeddings + UMAP visualisation + similarity index
# ---------------------------------------------------------------------------


def compute_embeddings(dataset: "fo.Dataset") -> None:
    """Compute CLIP embeddings, UMAP visualisation, and similarity index.

    Uses FiftyOne Brain:
    - ``fob.compute_visualization`` — 2-D UMAP projection of CLIP embeddings,
      stored under the key ``clip_umap`` (visible as a panel in the App).
    - ``fob.compute_similarity``    — brute-force nearest-neighbour index so
      you can run 'find visually similar images' queries in the App.

    Dependencies (all in pyproject.toml):
        open-clip-torch, umap-learn, fiftyone[brain]
    """
    try:
        import fiftyone.brain as fob  # deferred
    except ImportError:
        print(
            "[embeddings] WARNING: fiftyone.brain not available.\n"
            "  Install with:  uv add fiftyone open-clip-torch umap-learn"
        )
        return

    # ---- UMAP visualisation ------------------------------------------------
    existing_brain_keys = dataset.list_brain_runs()

    if VISUALIZATION_KEY in existing_brain_keys:
        print(f"[embeddings] Visualization '{VISUALIZATION_KEY}' already computed — skipping.")
    else:
        print(f"[embeddings] Computing CLIP embeddings + UMAP projection (key='{VISUALIZATION_KEY}') ...")
        print("             This may take several minutes on CPU ...")
        try:
            fob.compute_visualization(
                dataset,
                model=CLIP_MODEL,
                brain_key=VISUALIZATION_KEY,
                method="umap",
                num_workers=4,
            )
            dataset.save()
            print(f"[embeddings] UMAP visualisation saved under key '{VISUALIZATION_KEY}'.")
        except Exception as exc:
            print(f"[embeddings] WARNING: UMAP visualisation failed: {exc}")
            print("             Continuing without UMAP.")

    # ---- Similarity index --------------------------------------------------
    if SIMILARITY_KEY in existing_brain_keys:
        print(f"[embeddings] Similarity index '{SIMILARITY_KEY}' already computed — skipping.")
    else:
        print(f"[embeddings] Building CLIP similarity index (key='{SIMILARITY_KEY}') ...")
        try:
            fob.compute_similarity(
                dataset,
                model=CLIP_MODEL,
                brain_key=SIMILARITY_KEY,
            )
            dataset.save()
            print(f"[embeddings] Similarity index saved under key '{SIMILARITY_KEY}'.")
        except Exception as exc:
            print(f"[embeddings] WARNING: Similarity index failed: {exc}")
            print("             Continuing without similarity index.")


# ---------------------------------------------------------------------------
# 4. Image quality analysis
# ---------------------------------------------------------------------------


def _try_plugin_quality(dataset: "fo.Dataset") -> bool:
    """Attempt to run quality operators via the image-quality FiftyOne plugin.

    Returns True if successful, False if the plugin is not installed or fails.
    The plugin is: https://github.com/jacobmarks/image-quality-issues

    Install with:
        uv run fiftyone plugins download https://github.com/jacobmarks/image-quality-issues
    """
    try:
        import fiftyone.operators as foo  # deferred
        import fiftyone.plugins as fop    # deferred

        # Plugin is registered as @jacobmarks/image_issues (note: underscore)
        plugin_name = "@jacobmarks/image_issues"
        installed = [p.name for p in fop.list_plugins()]
        if plugin_name not in installed:
            print(
                f"[quality] {plugin_name} plugin is not installed.\n"
                "  Install it with:\n"
                "    uv run fiftyone plugins download "
                "https://github.com/jacobmarks/image-quality-issues"
            )
            return False

        # Operators exposed by the plugin (full URI: <plugin>/<operator>)
        plugin_operators = [
            f"{plugin_name}/compute_brightness",
            f"{plugin_name}/compute_blurriness",
            f"{plugin_name}/compute_contrast",
            f"{plugin_name}/compute_entropy",
            f"{plugin_name}/compute_exposure",
            f"{plugin_name}/compute_aspect_ratio",
            f"{plugin_name}/compute_saturation",
        ]

        ran_any = False
        for op_uri in plugin_operators:
            try:
                operator = foo.get_operator(op_uri)
                print(f"[quality]   Running {op_uri} ...")
                # Plugin operators expose __call__(sample_collection, ...) for
                # programmatic invocation — see jacobmarks/image-quality-issues
                operator(dataset)
                ran_any = True
            except Exception as exc:
                print(f"[quality]   WARNING: {op_uri} failed: {exc}")

        if not ran_any:
            return False

        dataset.save()
        print("[quality] Plugin operators completed.")
        return True

    except Exception as exc:
        print(f"[quality] Plugin execution failed ({exc}), falling back to manual metrics.")
        return False


def _manual_quality_metrics(dataset: "fo.Dataset") -> None:
    """Compute quality metrics manually using PIL + numpy as a fallback.

    Fields added to each sample:
    - ``quality_brightness``  — mean grayscale pixel value [0, 255]
    - ``quality_blurriness``  — Laplacian-edge variance (lower = blurrier)
    - ``quality_aspect_ratio``— width / height
    """
    try:
        import numpy as np
        from PIL import Image, ImageFilter  # deferred
    except ImportError:
        print(
            "[quality] WARNING: Pillow / numpy not installed — cannot compute manual metrics.\n"
            "  Install with:  uv add Pillow numpy"
        )
        return

    print(f"[quality] Computing manual quality metrics for {len(dataset)} samples ...")
    updated = 0

    for sample in dataset.iter_samples(progress=True, autosave=True):
        try:
            img = Image.open(sample.filepath).convert("RGB")
            gray = img.convert("L")
            arr = np.asarray(gray, dtype=np.float32)

            # Brightness: mean pixel intensity
            brightness = float(arr.mean())

            # Blurriness: variance of Laplacian (edge detection).
            # PIL.ImageFilter.Kernel requires `size` as a (width, height) tuple.
            laplacian = gray.filter(ImageFilter.Kernel(
                size=(3, 3),
                kernel=[-1, -1, -1,
                        -1,  8, -1,
                        -1, -1, -1],
                scale=1,
                offset=0,
            ))
            lap_arr = np.asarray(laplacian, dtype=np.float32)
            blurriness = float(lap_arr.var())

            # Aspect ratio
            w, h = img.size
            aspect_ratio = w / h if h > 0 else 0.0

            sample["quality_brightness"] = brightness
            sample["quality_blurriness"] = blurriness
            sample["quality_aspect_ratio"] = aspect_ratio
            updated += 1

        except Exception as exc:
            print(f"[quality]   WARNING: Could not process {sample.filepath}: {exc}")

    print(f"[quality] Manual metrics computed for {updated}/{len(dataset)} samples.")


def compute_image_quality(dataset: "fo.Dataset") -> None:
    """Run image-quality analysis, preferring the FiftyOne plugin.

    Falls back to manual PIL/numpy metrics if the plugin is unavailable.
    """
    print("[quality] Starting image quality analysis ...")

    # Check if plugin metrics already exist (sample the first sample)
    sample = dataset.first()
    if sample is not None and (
        hasattr(sample, "brightness") or
        hasattr(sample, "quality_brightness")
    ):
        print("[quality] Quality fields already present on dataset — skipping.")
        return

    plugin_success = _try_plugin_quality(dataset)

    if not plugin_success:
        print("[quality] Running fallback manual quality metrics ...")
        _manual_quality_metrics(dataset)

    dataset.save()
    print("[quality] Image quality analysis complete.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=" * 60)
    print("  FiftyOne Vegetables Detector Workflow")
    print("=" * 60)

    # Step 1: Download dataset
    dataset_dir = download_dataset()

    # Step 2: Build (or load) FiftyOne dataset
    import fiftyone as fo  # noqa: E402 — imported here so errors surface clearly
    dataset = build_fiftyone_dataset(dataset_dir)

    # Step 3: Compute embeddings
    compute_embeddings(dataset)

    # Step 4: Image quality analysis
    compute_image_quality(dataset)

    print()
    print("=" * 60)
    print(f"  Dataset '{DATASET_NAME}' ready with {len(dataset)} samples.")
    print("  Launching FiftyOne App at http://localhost:5151 ...")
    print("  Press Ctrl-C to exit.")
    print("=" * 60)

    session = fo.launch_app(dataset)
    session.wait()
