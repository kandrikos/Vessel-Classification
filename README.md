# Underwater Acoustic Vessel Classification

This repository contains a deep learning model for classifying hydrophone recordings of vessels into different categories based on their acoustic signatures. The model uses a modified ResNet18 architecture with multiple spectral features as input.

## Project Overview

The goal of this project is to classify underwater acoustic recordings into 5 categories:
- 4 classes of boat types:
  - Tug
  - Passenger ship
  - Cargo
  - Tanker
- 1 class for background noise

The model uses spectrograms generated from underwater audio recordings with multiple preprocessing techniques combined as inputs to a CNN.

## Dataset

This project uses the [VTUAD (Vessel Type Underwater Acoustic Data)](https://ieee-dataport.org/documents/vtuad-vessel-type-underwater-acoustic-data) dataset. Specifically, the "inclusion_2000_exclusion_4000" subset is used, which contains recordings where:
- One vessel is within 2000 meters of the hydrophone
- No other vessels are within 4000 meters

The audio files are 1 second in duration and the dataset is pre-split into train/test/validation sets.

## Features

- **Multi-channel spectrogram input**: Combines three different acoustic preprocessing methods:
  - Mel spectrogram
  - Short-time Fourier transform (STFT)
  - Log-scaled spectrogram
- **Data augmentation**: Includes spectrogram masking techniques for better generalization
- **Modified ResNet18 architecture**: Adapted for acoustic classification tasks
- **Comprehensive evaluation**: Includes precision, recall, F1-score, and confusion matrix


## Project Structure

```
src/
├── data/                      
│   ├── data_generator.py     
│   ├── preprocessing_generator.py
│   └── preprocessing.py
├── models/               
│   ├── classifier.py 
│   └── spec_augmentation.py                       
├── utils/
│   └── evaluator.py
```

## Usage

### Data Preprocessing

Before training the model, you need to preprocess the raw audio files to generate spectrograms. This is a necessary first step:

```bash
# Generate spectrograms from raw audio files
python -m src.data.preprocessing_generator
```

This script will:
1. Read the raw `.wav` files from the dataset
2. Apply three different preprocessing methods to generate spectrograms:
   - Mel spectrogram (tfmel)
   - Short-time Fourier transform (stft)
   - Log-scaled spectrogram (logscale)
3. Save the resulting spectrograms as `.npy` files in the appropriate directory structure

Make sure to set the correct path to your dataset in the preprocessing script or configuration file. The default sample rate used is 32000 Hz.

### Expected Directory Structure

After preprocessing, your directory structure should look like this:

```
Datasets/
└── VTUAD/
    └── inclusion_2000_exclusion_4000/
        ├── train/
        │   ├── audio/                # Original raw audio files
        │   │   ├── tug/
        │   │   ├── tanker/
        │   │   ├── cargo/
        │   │   ├── passengership/
        │   │   └── background/
        │   ├── tfmel/                # Generated Mel spectrograms
        │   │   ├── tug/
        │   │   ├── tanker/
        │   │   ├── cargo/
        │   │   ├── passengership/
        │   │   └── background/
        │   ├── stft/                 # Generated STFT spectrograms
        │   │   └── ...
        │   └── logscale/             # Generated log-scaled spectrograms
        │       └── ...
        ├── validation/
        │   └── ...
        └── test/
            └── ...
```

The preprocessing script will automatically create the necessary directories if they don't exist.

### Training and Evaluation

After preprocessing is complete, you can train and evaluate the model using YAML configuration files:

```bash
python main.py --config config/vessel_classification.yaml
```

Alternatively, run with default settings:

```bash
python main.py
```

#### Configuration Files

The project uses YAML configuration files to make it easier to experiment with different settings. Example configuration files are provided in the `config/` directory:

- `vessel_classification.yaml`: Configuration settings

You can create your own configuration files by modifying these examples. The configuration files allow you to specify:

- Data parameters (preprocessing methods, batch size)
- Model architecture parameters (filters, kernel sizes, regularization)
- Optimizer settings (type, learning rate, momentum)
- Training parameters (epochs, early stopping, learning rate scheduling)
- Output directories

This will:
1. Load the preprocessed spectrogram data
2. Initialize the vessel classifier with the ResNet18 architecture
3. Train the model with early stopping and learning rate reduction
4. Evaluate the model on the test set
5. Generate performance reports and visualizations

## Model Architecture

The model is based on the ResNet18 architecture with some modifications:
- Input shape: (95, 126, 3) - representing time-frequency spectrograms with 3 channels
- 3 input channels for different spectrogram types
- Added dropout for regularization
- L2 regularization to prevent overfitting
- Optional spectrogram augmentation

The model has approximately 11 million trainable parameters.

## Results

The model achieves excellent performance on the test set:

- **Test Accuracy**: 98.22%
- **F1-Score**: 97.36%
- **Precision**: 96.75%
- **Recall**: 98.06%

### Class-specific Performance

| Class         | Precision | Recall | F1-score | Support |
|---------------|-----------|--------|----------|---------|
| Tug           | 99.08%    | 97.07% | 98.07%   | 445     |
| Tanker        | 90.30%    | 98.90% | 94.40%   | 94      |
| Cargo         | 99.00%    | 97.00% | 98.30%   | 627     |
| Passengership | 97.10%    | 97.10% | 97.10%   | 35      |
| Background    | 98.00%    | 99.70% | 98.80%   | 655     |

### Confusion Matrix

The confusion matrix shows that the model performs well across all classes, with very few misclassifications.

## Training Process

The model is trained with:
- SGD optimizer with momentum and Nesterov acceleration
- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- Spectrogram augmentation for improved generalization

## Citations

If you use this code or the VTUAD dataset, please cite the original paper:

```
@INPROCEEDINGS{9940921,
  author={Phan, Huy and Andreev, Alexey and Kulik, Denys and Inoue, Tatsuya and Koch, Paul and Mitsufuji, Yuki},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Underwater Acoustic Vessel Type Classification},
  year={2023},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10095239}
}
```

## Acknowledgements

- VTUAD dataset creators for providing the underwater acoustic data