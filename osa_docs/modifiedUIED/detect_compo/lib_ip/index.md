# lib_ip

## Overview

The `lib_ip` module is a comprehensive image processing and UI component detection library designed to identify, analyze, and extract visual elements from screenshots. It provides a complete pipeline for detecting rectangular components (buttons, text fields, images, etc.) within GUI interfaces, performing spatial analysis on detected elements, and managing their geometric relationships. The module combines binary image processing, morphological operations, and boundary detection algorithms to extract meaningful UI components from raw image data.

## Purpose

This module serves as the core image analysis engine for the TEEPT GUI Assistant system, enabling the identification and characterization of interactive UI elements within desktop screenshots. Specifically, it:

- **Detects UI Components**: Identifies rectangular and non-rectangular visual elements in binary images through boundary analysis and connected component detection, extracting their precise spatial coordinates and dimensions.

- **Processes and Filters Components**: Removes noise, filters components based on size and aspect ratio constraints, and eliminates irrelevant elements such as header/footer bars and thin lines that do not represent actionable UI components.

- **Analyzes Spatial Relationships**: Determines containment hierarchies and intersection relationships between components, enabling the system to understand nested UI structures and component groupings.

- **Categorizes Components**: Classifies detected elements into meaningful categories (Block, Image, Text, Noise) based on their geometric properties and content characteristics, facilitating targeted processing of different UI element types.

- **Merges and Consolidates Elements**: Combines overlapping or adjacent components (particularly text elements) into cohesive units, reducing fragmentation and improving the accuracy of component representation.

- **Extracts Nested Components**: Detects rectangular sub-components within image regions and performs recursive analysis to identify complex nested UI structures.

- **Provides Visualization and Export**: Renders detected components with bounding boxes and classifications onto images, and exports component data in multiple formats (CSV, JSON) for downstream processing and integration with the task automation system.

By performing these operations, the module transforms raw screenshots into structured, analyzable representations of UI layouts, enabling the TEEPT system to match user descriptions to specific interactive elements and provide accurate guidance for task completion.