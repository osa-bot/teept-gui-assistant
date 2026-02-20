# Component Detection Module

## Overview

The Component Detection module is a specialized image processing pipeline designed to automatically identify, extract, and classify UI components from desktop screenshots. It serves as the core visual analysis engine that transforms raw image data into structured representations of interactive UI elements. The module implements a comprehensive workflow that combines binary image processing, morphological operations, boundary detection, and spatial analysis to detect rectangular and non-rectangular visual elements within GUI interfaces.

## Purpose

This module is specifically designed to handle the screenshot analysis phase of the TEEPT GUI Assistant system. Its primary responsibilities include:

- **UI Component Extraction**: Identifies and extracts rectangular UI components (buttons, text fields, images, input areas, etc.) from binary image representations through boundary analysis and connected component detection algorithms.

- **Component Filtering and Refinement**: Removes noise and irrelevant visual elements such as thin lines, header/footer bars, and small artifacts that do not represent actionable UI components. Applies size and aspect ratio constraints to ensure only meaningful components are retained.

- **Spatial Relationship Analysis**: Determines hierarchical containment relationships and intersection patterns between detected components, enabling the system to understand nested UI structures and component groupings within complex layouts.

- **Component Classification**: Categorizes detected elements into meaningful types (Block, Image, Text, Noise) based on their geometric properties and content characteristics, facilitating targeted processing of different UI element categories.

- **Component Consolidation**: Merges overlapping or adjacent components, particularly fragmented text elements, into cohesive units to reduce fragmentation and improve component representation accuracy.

- **Nested Structure Detection**: Performs recursive analysis to identify and extract sub-components within larger image regions, capturing complex nested UI hierarchies.

- **Data Conversion and Export**: Transforms detected component objects into structured array representations containing spatial coordinates, dimensions, categories, and identifiers for downstream processing and integration with the task automation system.

By performing these operations, the module converts raw screenshots into analyzable representations of UI layouts, enabling the TEEPT system to accurately match user descriptions to specific interactive elements and provide precise guidance for task completion.