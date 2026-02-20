# Result Processing

## Overview

The Result Processing module is responsible for evaluating and analyzing detection results from UI component detection systems against ground truth annotations. It provides comprehensive tools for loading detection outputs and ground truth data from JSON files, computing performance metrics using Intersection over Union (IoU) and Intersection over Detection (IoD) calculations, and visualizing results through bounding box rendering on images.

## Purpose

This module serves as the evaluation backbone for the TEEPT GUI Assistant's UI component detection pipeline. Its primary functions are:

- **Loading and Parsing Detection Data**: Reads JSON-formatted detection results from component detection systems and reformats them into structured dictionaries indexed by image identifiers, with support for filtering components based on size and position criteria.

- **Loading Ground Truth Annotations**: Processes COCO-format JSON files containing ground truth UI component annotations, organizing them by image with bounding box coordinates, category mappings, and image dimensions.

- **Performance Evaluation**: Compares detected UI components against ground truth annotations across multiple images, calculating precision, recall, and F1 scores. Supports filtering evaluations by component type (text versus non-text components) and categorization by component size ranges.

- **Visualization and Debugging**: Provides functionality to draw bounding boxes on images for visual inspection of both detection results and ground truth annotations, enabling manual verification of detection accuracy and identification of systematic errors.

- **Coordinate Transformation**: Implements bounding box resizing and scaling operations to handle cases where detection and ground truth data originate from images of different dimensions, ensuring fair comparison through height-ratio-based normalization.

The module enables quantitative assessment of detection system performance and provides visual tools for qualitative analysis, supporting iterative improvement of UI component detection accuracy within the TEEPT GUI Assistant system.