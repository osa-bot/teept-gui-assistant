# UIED Config

## Overview

The UIED Config module provides centralized configuration management for the UI element detection and classification pipeline. It establishes the foundational setup required for machine learning-based UI component analysis, including model initialization, directory structure organization, and classification mappings. This module serves as the single source of truth for all configuration parameters, thresholds, and visual properties used throughout the UI detection workflow.

## Purpose

This module is designed to:

- **Initialize machine learning models**: Configure paths and parameters for CNN and EAST models used in UI element detection and text localization
- **Manage directory structures**: Organize input/output directories for processing pipelines, including paths for original images, OCR results, merged components, and extracted UI elements
- **Define classification mappings**: Establish lookup tables that map numeric class indices to specific UI component types (buttons, text fields, images, etc.)
- **Standardize detection thresholds**: Centralize geometric and visual thresholds for shape detection, text analysis, and component classification across the detection pipeline
- **Provide visualization configuration**: Define color schemes for rendering detected UI components and classification labels during visual feedback and debugging
- **Establish image processing parameters**: Configure image dimensions and preprocessing specifications required for model inference

By consolidating all configuration settings in a single module, this component ensures consistency across the UI element detection system and simplifies parameter management for different processing stages, from initial screenshot capture through final UI component extraction and classification.