# MI-IWNet

MI-IWNet is a research codebase for oceanic internal wave segmentation from
multi-source satellite imagery. The repository focuses on joint training and
inference for SAR and RGB data, with multiple segmentation backbones, shared
multi-modal design choices, and a collection of analysis scripts used for
figure generation and case studies.

This repository is best understood as a code release rather than a packaged
benchmark. Large datasets are not included, and several paths in the training
and inference scripts are written for the author's remote Linux environment.

## Highlights

- Joint SAR and RGB segmentation workflow.
- Shared multi-modal training pipeline with alternating modality batches.
- Support for multiple backbone families, including ConvNeXt-based U-Net,
  UNet++, SegFormer, TransUNet, SwinUNet, IWResNet, IWENet, and MTU2Net.
- Patch-based training and inference for large satellite images.
- Dynamic filtering of sparse RGB patches during training.
- Training logs, checkpoints, and metric curves saved automatically.

## Repository Structure

```text
MI-IWNet/
|-- configs/        Training configuration
|-- models/         Segmentation models and backbone modules
|-- train/          Training entrypoint, loops, logging, and loaders
|-- utils/          Dataset classes, transforms, and data preparation
|-- data_utils/     Inference, visualization, and helper scripts
|-- analysis/       Analysis code, generated figures, and case-study outputs
|-- .gitignore
`-- README.md
```

Core directories:

- `configs/`: central experiment settings in `configs/train_config.py`
- `train/`: training entrypoint in `train/main.py`
- `utils/`: paired-data preparation and dataset loading
- `data_utils/`: standalone SAR and RGB inference scripts
- `analysis/`: supplementary scripts and outputs; not required for model
  training

## Data Layout

The default configuration expects data under a Linux-style root such as:

```text
/home/.../internal_wave_detection_project/IW_data/
|-- images_sar/
|-- masks_sar/
|-- images_rgb/
`-- masks_rgb/
```

Important notes:

- Image files and mask files are matched by filename stem.
- `utils/prepare_data.py` automatically pairs files and splits them into
  train, validation, and test subsets.
- By default, the split ratio in `configs/train_config.py` is `0.7 / 0.2 / 0.1`.
- During preparation, paired samples are copied into:

```text
DATA_ROOT/
|-- SAR/
|   |-- train/
|   |-- val/
|   `-- test/
`-- RGB/
    |-- train/
    |-- val/
    `-- test/
```

### Modality Notes

- SAR inputs may be `jpg`, `png`, `tif`, or `tiff`.
- SAR JPG files should be treated as plain imagery without reliable
  georeferencing.
- SAR TIFF files may contain useful geographic extent in TIFF tags even when
  CRS or affine transform metadata is not meaningful.
- RGB TIFF files in the author's dataset are treated as imagery inputs rather
  than as per-file georeferenced tiles.

## Environment Setup

There is no pinned `requirements.txt` in this repository yet, so dependencies
need to be installed manually.

Recommended baseline environment:

- Python 3.8+
- PyTorch with CUDA support if you plan to train on GPU

Typical packages used by the current code:

```bash
pip install numpy opencv-python matplotlib scikit-learn albumentations tifffile tqdm lion-pytorch timm
```

Some backbone components also reference OpenMMLab libraries. If you use those
variants, install versions of `mmcv` and related packages that are compatible
with your PyTorch and CUDA stack.

## Configuration

Before training or inference, review `configs/train_config.py`.

Key fields include:

- `DATA_ROOT`
- `SAR_IMAGE_DIR`
- `RGB_IMAGE_DIR`
- `SAR_MASK_DIR`
- `RGB_MASK_DIR`
- `MODEL_TYPE`
- `IMG_SIZE`
- `BATCH_SIZE`
- `NUM_EPOCHS`
- `LOG_SAVE_DIR`
- `PTH_SAVE_DIR`
- `FIG_SAVE_DIR`

Supported model names listed in the config include:

```text
unet, ConvNeXt, UNetPlusPlus, SegFormer, TransUNet,
IWResNet, SwinUNet, IWENet, mtu2net
```

## Training

1. Edit dataset paths and output paths in `configs/train_config.py`.
2. Select the desired `MODEL_TYPE`.
3. Start training:

```bash
python train/main.py
```

What the training pipeline currently does:

- pairs SAR and RGB images with masks by filename stem
- creates train/val/test splits
- loads SAR and RGB through dedicated dataset logic
- filters low-positive RGB patches during training
- trains with alternating modality batches
- logs metrics including IoU, Dice, Precision, Recall, Accuracy, and BF Score

Outputs are saved under the directories specified in the config, such as
checkpoints, logs, metric curves, and best-model weights.

## Inference

Two standalone scripts are provided:

- `data_utils/inference_sar.py`
- `data_utils/inference_rgb.py`

Before running them, update the hardcoded paths inside each script:

- model checkpoint path
- input image directory
- output directory

Run SAR inference:

```bash
python data_utils/inference_sar.py
```

Run RGB inference:

```bash
python data_utils/inference_rgb.py
```

The scripts save predicted masks and visualization overlays for each input
image.

## Analysis Folder

The `analysis/` directory contains figure-generation scripts, region-specific
case-study outputs, and other research artifacts. These files are useful for
reproducing visual analysis, but they are not required for the core training
pipeline in `train/` and `utils/`.

## Current Limitations

- Dataset paths are still configured for a personal remote environment.
- Dependency installation is manual because the repository does not yet include
  a pinned environment file.
- Some scripts use hardcoded input and output paths.
- This repository includes research outputs alongside reusable training code,
  so not every file is part of the minimal training workflow.

## Suggested Next Steps

If you plan to keep maintaining this repository, the most helpful follow-up
cleanup items are:

1. add a `requirements.txt` or `environment.yml`
2. move hardcoded paths into config files or CLI arguments
3. further tighten `.gitignore` for generated figures, caches, and temporary
   outputs
