# Text Detection Module

## Overview

The Text Detection module is a comprehensive OCR (Optical Character Recognition) processing system that detects, extracts, and processes text from images. It serves as a critical component in the UI element identification pipeline by converting raw image data into structured text information with precise spatial coordinates. The module integrates multiple OCR backends (Google Cloud Vision and PaddleOCR) and applies sophisticated post-processing techniques to deliver clean, merged, and contextually meaningful text detection results.

## Purpose

This module is designed to fulfill the text-based UI element detection requirements of the TEEPT GUI Assistant system. Specifically, it:

- **Performs OCR Analysis**: Executes optical character recognition on input images using either Google Cloud Vision API or PaddleOCR, extracting text content and bounding box coordinates from detected regions.

- **Standardizes OCR Output**: Converts detection results from different OCR backends (Google and PaddleOCR formats) into a unified Text object representation with normalized location data, ensuring consistent downstream processing regardless of the OCR method used.

- **Enhances Text Quality**: Applies noise filtering to remove spurious single-character detections and merges intersected text regions to consolidate fragmented detections into coherent units.

- **Reconstructs Sentences**: Combines separately detected words into complete sentences, improving the semantic quality of extracted text for more accurate UI element matching and description comparison.

- **Generates Structured Output**: Converts processed text detection results into JSON-serializable format containing image metadata, text content, and precise bounding box coordinates for integration with the broader UI analysis pipeline.

- **Provides Visual Feedback**: Visualizes detected text regions on the original image with optional display and file output capabilities, enabling validation and debugging of detection results.