# CNN Module

## Overview

The CNN module is a comprehensive image classification system built on Convolutional Neural Networks. It provides end-to-end functionality for loading image datasets, building and training CNN models using transfer learning, and performing inference on new images. The module integrates data management, model architecture, and configuration handling into a cohesive framework designed for multi-class image classification tasks.

## Purpose

This module serves as the image analysis backbone for the TEEPT GUI Assistant system. It enables the detection and classification of UI elements through visual analysis by:

- **Data Management**: Loading and preprocessing image datasets from disk, organizing them by classification categories, and preparing normalized training and testing splits with proper label encoding for model training.

- **Model Architecture and Training**: Building transfer learning-based CNN models using ResNet50 as the backbone, training these models on prepared datasets, and persisting trained models to disk for later use.

- **Configuration Management**: Centralizing configuration parameters including dataset paths, model file locations, class mappings, and image dimensions to ensure consistent operation across the system.

- **Image Inference**: Preprocessing new images to match model input requirements and performing predictions on single or multiple images to classify UI elements detected in screenshots.

- **Model Evaluation**: Assessing trained model performance on test datasets and computing performance metrics to validate classification accuracy.

The module supports multiple classifier types (Text, Noise, Elements, and Image classifiers) enabling specialized detection capabilities for different aspects of GUI analysis within the broader task automation workflow.