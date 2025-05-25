# Multimodal Malware Classification System

![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/license-MIT-green)

This repository contains the implementation of a multimodal deep learning system for malware classification using tabular features, images, and API call sequences.
Code can also be aссessed via [Google Colab](https://colab.research.google.com/drive/1vQCmxRed8WOzNk1ghHaNpNTxsQqYT8Jv?usp=sharing)

## Table of Contents
- [Project Description](#project-description)
- [Features](#features)
- [Data](#data)
- [Command Line Arguments](#command-line-arguments)
- [Results](#results)
- [License](#license)
- [Citing](#citing)

  
## Project Description

This project was developed as part of my Master's thesis in Computer Science. The system combines three different neural network architectures to process different malware representations:

1. **Tabular Data**: Deep Neural Network (DNN) for processing static analysis features
2. **Image Data**: Convolutional Neural Network (CNN) for processing malware visualizations
3. **Sequence Data**: DistilBERT model for processing API call sequences

The model achieves state-of-the-art performance on the [Multimodal CIC-AndMal2020 dataset](https://www.unb.ca/cic/datasets/andmal2020.html).
  
## Features

- **Multimodal Architecture**: Processes 3 different data types simultaneously
- **State-of-the-art Models**: Uses CNN for images, DNN for tabular data, and DistilBERT for sequences
- **Memory Efficient**: Optimized for training on single GPU
- **Reproducible Research**: Complete experiment tracking

## Data

The model checkpoint checkpoint_epoch_15.pth is available on [Google Drive](https://drive.google.com/file/d/1jpPi7tZFKmliiZHG7YH1ptaFhnoXmH2K/view?usp=sharing).

## Command Line Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--tabular_path` | /Tabular | Path to tabular data |
| `--image_path` | /Image | Path to image data |
| `--sequence_path` | /Sequence | Path to sequence data |
| `--epochs` | 15 | Number of training epochs |
| `--batch_size` | 32 | Training batch size |
| `--learning_rate` | 1e-4 | Initial learning rate |
| `--device` | cuda | Device to use (cuda/cpu) |

## Results

Multimodal approach achieves the following performance:

|    Model   | Accuracy | Precision | Recall | F1-Score |
|------------|----------|-----------|--------|----------|
| Multimodal | 0.95     | 0.94      | 0.94   | 0.94     |

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Citing

@mastersthesis{
  author = {ElenaSindiukova},
  title = {Multimodal Analysis in Cybersecurity},
  school = {HSE, Moscow},
  year = {2025}
}
