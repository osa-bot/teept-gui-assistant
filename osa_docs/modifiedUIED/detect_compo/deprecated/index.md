# Deprecated Module

## Overview

The Deprecated module provides foundational image processing and UI component analysis utilities for the TEEPT GUI Assistant system. It contains core functionality for segmenting images, detecting and organizing UI blocks, classifying text regions, and managing hierarchical block structures. This module serves as a backend processing layer that prepares raw screenshots for higher-level UI element detection and analysis.

## Purpose

This module is designed to handle the initial stages of screenshot analysis and preparation:

- **Image Segmentation**: Divides large screenshots into manageable horizontal strips with configurable overlap, enabling efficient processing of full-screen captures that may exceed memory or processing constraints.

- **Boundary Clipping and Region Extraction**: Clips image regions using horizontal and vertical dividing lines, extracting and organizing rectangular sub-regions from complex layouts. Supports both hollow background generation (transparent component regions) and filled background generation (color-sampled component regions).

- **Block Detection and Organization**: Identifies rectangular layout blocks within images and establishes hierarchical parent-child relationships between blocks based on their compositional containment. Provides mechanisms to classify blocks as UI components or navigation bars, and enables removal of block regions from binary image representations.

- **Text Region Classification**: Analyzes image regions using OCR (Tesseract) to detect and classify text content, determining whether regions contain meaningful text based on area thresholds and filtering out small or invalid detections.

- **Binary Image Manipulation**: Supports erasing detected block regions from binary image maps with optional padding, facilitating iterative detection workflows where processed regions are removed for subsequent analysis passes.

The module operates on preprocessed image data and binary representations, providing the structural decomposition and classification necessary for the system to identify actionable UI components within desktop screenshots.