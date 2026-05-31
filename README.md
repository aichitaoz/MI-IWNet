# MI-IWNet

This repository contains the PyTorch implementation of `MI-IWNet`, a
multi-source internal wave segmentation framework for SAR and RGB satellite
imagery. The codebase accompanies the corresponding research manuscript and
includes training, inference, and analysis scripts used in the study.

## Overview

MI-IWNet is designed for oceanic internal wave extraction under heterogeneous
imaging conditions. The current repository supports:

- joint SAR and RGB training
- alternating multi-modal optimization
- patch-based segmentation for large images
- multiple comparative backbones
- analysis scripts used for figure generation and case studies

The repository should be viewed as a research code release. Large-scale
datasets are not included in this public version.

## Repository Structure

```text
MI-IWNet/
|-- configs/        experiment configuration
|-- models/         network definitions and backbone modules
|-- train/          training entrypoints and optimization loops
|-- utils/          dataset preparation and loading utilities
|-- data_utils/     inference and visualization scripts
|-- analysis/       analysis code and supplementary outputs
`-- README.md
```

Main components:

- `train/main.py`: training entrypoint
- `configs/train_config.py`: dataset paths and experiment settings
- `utils/prepare_data.py`: image-mask pairing and data split
- `utils/InterWaveDataset.py`: patch-based dataset loader
- `data_utils/inference_sar.py`: SAR inference script
- `data_utils/inference_rgb.py`: RGB inference script

## Models

The repository currently includes the following model options in
`configs/train_config.py`:

```text
unet, ConvNeXt, UNetPlusPlus, SegFormer, TransUNet,
IWResNet, SwinUNet, IWENet, mtu2net
```

These models are used either as the main framework components or as comparison
baselines in the experimental workflow.

## Data

The present code assumes a local or remote dataset root defined in
`configs/train_config.py`. By default, SAR and RGB images are organized
separately and paired with their segmentation masks by filename stem.

Expected directory pattern:

```text
DATA_ROOT/
|-- images_sar/
|-- masks_sar/
|-- images_rgb/
`-- masks_rgb/
```

After pairing, `utils/prepare_data.py` splits the data into train, validation,
and test subsets according to the configured ratios.

The original research data are not distributed in this repository.

## Training

Before training, modify the dataset and output paths in
`configs/train_config.py`, then run:

```bash
python train/main.py
```

The current training pipeline includes paired data preparation, patch-based
loading, alternating SAR/RGB batches, and checkpoint saving.

## Inference

Two standalone inference scripts are provided:

```bash
python data_utils/inference_sar.py
python data_utils/inference_rgb.py
```

Please update the checkpoint path and input/output directories in the scripts
before execution.

## Environment

This repository does not yet provide a fixed `requirements.txt`. The current
codebase mainly depends on:

- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- Albumentations
- scikit-learn
- tifffile
- tqdm
- lion-pytorch
- timm

Some backbone implementations may additionally require OpenMMLab-related
packages such as `mmcv`.

## Analysis Scripts

The `analysis/` directory contains scripts and generated outputs used in the
manuscript-level analysis, including regional case studies, visualization, and
figure preparation. These files are supplementary to the core training and
inference pipeline.

## Citation

If you use this repository in academic work, please cite the corresponding
paper. The formal citation can be added here after publication.

```bibtex
@article{mi_iwnet,
  title   = {MI-IWNet},
  author  = {},
  journal = {},
  year    = {}
}
```
