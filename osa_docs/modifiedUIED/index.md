# Modified UIED Module

## Overview

The Modified UIED (UI Element Detection) module is a comprehensive image analysis system designed to detect, extract, and classify UI components from desktop screenshots. It serves as the core visual analysis engine that transforms raw image data into structured representations of interactive UI elements through a multi-stage detection pipeline. The module integrates text detection via OCR, component boundary analysis, and intelligent result merging to provide complete UI element identification with spatial coordinates and categorical classifications.

## Purpose

This module fulfills the screenshot analysis and UI element detection requirements of the TEEPT GUI Assistant system. Specifically, it:

- **Executes Complete UI Detection Workflows**: Runs the full UIED pipeline on input images, performing sequential text detection, component detection, and result merging to produce comprehensive UI element inventories with spatial and categorical information.

- **Performs Text-Based Element Detection**: Integrates OCR processing to extract text content and bounding box coordinates from images, supporting multiple OCR backends (Google Cloud Vision and PaddleOCR) with unified output formatting and quality enhancement through noise filtering and text reconstruction.

- **Detects Visual UI Components**: Identifies and extracts rectangular and non-rectangular UI components from binary image representations through boundary analysis, morphological operations, and connected component detection, filtering out noise and irrelevant visual elements.

- **Classifies Detected Elements**: Categorizes detected components into meaningful types (Block, Image, Text, Noise) based on geometric properties and content characteristics, enabling specialized processing of different UI element categories.

- **Merges Detection Results**: Integrates separate detection outputs from text recognition and component analysis systems into cohesive UI element representations, establishing hierarchical relationships, eliminating redundancies, and refining results through intelligent filtering.

- **Manages Configuration and Models**: Centralizes configuration parameters, machine learning model initialization (CNN and EAST models), directory structures, and classification mappings to ensure consistent operation across all detection stages.

- **Evaluates Detection Performance**: Provides comprehensive evaluation tools for assessing detection accuracy against ground truth annotations, computing precision, recall, and F1 scores using Intersection over Union calculations, and visualizing results through bounding box rendering.

- **Supports Image Preprocessing**: Calculates appropriate image dimensions for processing while maintaining aspect ratios, enabling consistent model inference across images of varying sizes.

By performing these operations, the module converts raw screenshots into analyzable representations of UI layouts, enabling the TEEPT system to accurately identify interactive elements and provide precise guidance for task completion.