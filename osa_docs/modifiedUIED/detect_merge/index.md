# Detect Merge Module

## Overview

The Detect Merge module is responsible for post-processing and unifying detection results from multiple UI analysis sources. It provides a comprehensive framework for managing spatial elements, merging detection outputs, and refining detected components through hierarchical relationship establishment and intelligent filtering. The module serves as the integration layer between raw detection outputs (composition detection and text recognition) and the unified UI element representation required by downstream task automation components.

## Purpose

This module fulfills several critical functions within the TEEPT GUI Assistant pipeline:

**Element Representation and Management**: The module defines a unified spatial element abstraction that encapsulates bounding box coordinates, categorical classification, text content, and hierarchical relationships. This standardized representation enables consistent handling of detected UI components throughout the system.

**Detection Result Integration**: The module merges separate detection outputs from composition analysis and text recognition systems into a cohesive set of UI elements. This integration process handles coordinate alignment, eliminates redundancies, and establishes parent-child relationships between overlapping elements.

**Element Refinement and Filtering**: The module implements multiple refinement strategies to improve detection quality, including:
- Removal of noise through text content filtering based on length and size constraints
- Elimination of UI chrome elements (top and bottom bars) that are not relevant to task automation
- Filtering of elements contained within text regions to avoid duplicate representations
- Merging of adjacent text lines into coherent paragraph units for improved text element organization

**Spatial Analysis and Relationship Establishment**: The module provides utilities for calculating spatial relationships between elements, including intersection area computation, containment detection, and hierarchical parent-child relationship establishment based on spatial containment.

**Output Serialization and Visualization**: The module supports saving merged detection results to JSON format with complete element information and image dimensions, as well as visualization capabilities for debugging and validation of detection and merging processes.