# Brain MRI Segmentation using Dense Networks

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains a PyTorch implementation of dense network-based models for brain MRI segmentation tasks, including hemisphere segmentation and complete brain structure segmentation.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Input Data Format](#input-data-format)
- [Output](#output)
- [Model Architecture](#model-architecture)
- [Configuration Parameters](#configuration-parameters)
- [Image Preprocessing](#image-preprocessing)
- [Utilities](#utilities)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Notes](#notes)
- [Citation](#citation)
- [License](#license)

## Overview

The project implements two main segmentation tasks:
- **Hemisphere Segmentation**: Segments brain hemispheres from MRI scans.
- **Complete Brain Segmentation**: Performs multi-class segmentation of brain structures.

The models are based on DenseNet architecture with custom encoder-decoder designs optimized for medical image segmentation.

## Features

- Dense network architecture with skip connections.
- Support for both hemisphere and multi-class brain segmentation.
- DICOM to NIfTI conversion pipeline.
- Automatic orientation standardization.
- Memory-efficient inference on 3D volumes.
- Pretrained model support.

## Requirements

### System Requirements

The code was tested with the following environment:

| Requirement | Version |
|-------------|---------|
| Python      | **3.11** |
| PyTorch     | 2.x     |
| CUDA (optional) | 11.8 or 12.x |
| OS          | Linux / Mac / Windows |

⚠️ **Python 3.11 is REQUIRED. Other versions may fail to install dependencies.**

## Installation

### Prerequisites

#### Install Git

Git is required to clone the repository.

##### Windows
1. Download Git from the official website: [git-scm.com](https://git-scm.com/download/win)
2. Run the installer and follow the setup wizard.
3. During installation, you can accept the default settings.

##### Mac
- Using installer: Download from [git-scm.com](https://git-scm.com/download/mac)
- Or using Homebrew: `brew install git`

##### Linux (Ubuntu)
```bash
sudo apt update
sudo apt install git
```

Verify installation:
```bash
git --version
```

#### Install Python 3.11

Download Python 3.11 from the official website: [python.org](https://www.python.org/downloads/release/python-3110/).

##### Windows
1. Download: `python-3.11.x-amd64.exe`
2. Run the installer.
3. **Important**: During installation, check the box **"Add Python 3.11 to PATH"**.
4. If you forget to check it, you can manually add Python to PATH:
   - Search for "Environment Variables" in Windows search.
   - Click "Edit the system environment variables".
   - Click "Environment Variables".
   - Under "System variables", find "Path" and click "Edit".
   - Click "New" and add: `C:\Python311\` and `C:\Python311\Scripts\`
   - Click "OK" to save changes.

##### Mac
- Using installer: `python-3.11.x-macos.pkg`
- Or using Homebrew: `brew install python@3.11`

##### Linux (Ubuntu)
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

Verify Python installation:
```bash
python3.11 --version
```

### 2. Install CUDA (Optional but Recommended)

GPU acceleration significantly speeds up inference.

- Check if your GPU supports CUDA: [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus)
- Install CUDA from: [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
- Recommended version: CUDA 11.8

Verify installation:
```bash
nvcc --version
```

### 3. Clone the Repository
```bash
git clone https://github.com/usb1508/brain-mri-segmentation.git
cd brain-mri-segmentation
```

### 4. Create and Activate Python Environment

Creating a virtual environment prevents dependency conflicts.
```bash
python3.11 -m venv brain_env
```

Activate it:

#### Linux / Mac
```bash
source brain_env/bin/activate
```

#### Windows
```bash
brain_env\Scripts\activate
```

### 5. Install Dependencies

Upgrade pip first:
```bash
pip install --upgrade pip
```

Then install requirements:
```bash
pip install -r requirements.txt
```

### 6. Install PyTorch

If you **have a CUDA GPU**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

If you **do NOT have a GPU**:
```bash
pip install torch torchvision
```

## Usage

### Complete Brain Segmentation
```bash
python inference_complete.py --path_model CODE/model_DenseTV.pth --dir_data_src Data --dir_data_dst Data --num_classes 3
```

### Hemisphere Segmentation
```bash
python inference_hemi.py --path_model CODE/model_hemi_mask.pth --dir_data Data --dir_data_dst Data --num_classes 2
```

**Note**: Before running hemisphere segmentation, run the complete segmentation first. It may take a couple of minutes on a CPU-only environment.

## Input Data Format

The pipeline expects:
- **DICOM files**: Organized in subject/visit directory structure.
- **NIfTI files**: Pre-converted medical images.

**Directory structure**:
```
Data/
├── Subject1/
│   ├── Visit1/
│   │   ├── *.dcm (DICOM files)
│   │   └── *.nii.gz (NIfTI files)
│   └── Visit2/...
├── Subject2/
│   ├── Visit1/
│   │   ├── *.dcm (DICOM files)
│   │   └── *.nii.gz (NIfTI files)
│   └── Visit2/...
```

## Output

The inference scripts generate:
- `hemi_mask.nii.gz`: Hemisphere segmentation masks.
- `*_seg.nii.gz`: Complete brain segmentation masks.
- Standardized NIfTI files with canonical orientation.

## Model Architecture

The repository includes two main model variants:

### 1. Dense Model
- Used for hemisphere segmentation (2 classes).
- Encoder-decoder architecture with dense blocks.
- Skip connections between encoder and decoder.

### 2. DenseTV Model
- Used for complete brain segmentation (3+ classes).
- Enhanced decoder with additional feature fusion.
- Total variation regularization support.

### Core Classes

- **`_DenseLayer`**: Basic dense layer with batch normalization and dropout.
- **`_DenseBlock`**: Multiple dense layers with growth connections.
- **`Encoder`**: Feature extraction using dense blocks with downsampling.
- **`Decoder`/`DecoderTV`**: Upsampling decoder with skip connections.
- **`Dense`/`DenseTV`**: Complete segmentation models.

### Key Features

- **Memory Efficiency**: Optional memory-efficient dense layer computation.
- **Skip Connections**: Multi-scale feature fusion.
- **Pretrained Backbone**: Uses DenseNet-121 pretrained features.
- **Flexible Architecture**: Configurable block depths and growth rates.

## Configuration Parameters

### Model Parameters
- `block_config`: Tuple defining dense block configurations.
- `growth_rate`: Feature map growth rate in dense blocks (default: 32).
- `num_classes`: Number of segmentation classes.
- `pretrain`: Use pretrained DenseNet backbone.

### Inference Parameters
- `--path_model`: Path to trained model weights.
- `--dir_data`: Input data directory.
- `--dir_data_dst`: Output directory for results.

## Image Preprocessing

The pipeline includes:
1. **Orientation Standardization**: Convert to canonical orientation.
2. **Intensity Normalization**: Percentile-based normalization (99.7th percentile).
3. **Resizing**: Images resized to 448×448 for inference.
4. **ImageNet Normalization**: Standard normalization for pretrained features.

## Utilities

- **`checkdirctexist()`**: Directory creation utility.
- **`normalize()`**: Image normalization for model input.
- **`bias_field_correction()`**: N4 bias field correction using SimpleITK.
- **`convert_to_cbf()`**: Segmentation mask visualization.

## File Structure
```
├── model.py              # Model architectures
├── inference.py          # Core inference functions
├── inference_hemi.py     # Hemisphere segmentation script
├── inference_complete.py # Complete segmentation script
├── utils.py              # Utility functions
├── requirements.txt      # Dependencies
└── LICENSE               # License file
```

## Dependencies

Key dependencies include:
- PyTorch
- torchvision
- nibabel
- pydicom
- dicom2nifti
- SimpleITK
- opencv-python
- numpy
- matplotlib

## Notes

- The models expect GPU availability (CUDA) for optimal performance.
- DICOM files are automatically converted to NIfTI format if needed.
- Multi-planar DICOM series are filtered to keep only axial slices.
- Edge slices (first/last 2 slices) are excluded from segmentation.


## License

Please refer to the [LICENSE](LICENSE) file in the repository for usage terms.
