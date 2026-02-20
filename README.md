# teept-gui-assistant

---

![License](https://img.shields.io/github/license/andreygetmanov/teept-gui-assistant?style=flat&logo=opensourceinitiative&logoColor=white&color=blue)
[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

Built with:

![keras](https://img.shields.io/badge/Keras-D00000.svg?style=flat&logo=Keras&logoColor=white)
![numpy](https://img.shields.io/badge/NumPy-013243.svg?style=flat&logo=NumPy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458.svg?style=flat&logo=pandas&logoColor=white)
![tqdm](https://img.shields.io/badge/tqdm-FFC107.svg?style=flat&logo=tqdm&logoColor=black)

---

## Overview

Teept is an intelligent PC assistant that provides real-time, step-by-step guidance by analyzing screen content and highlighting relevant interface elements. It reduces user stress, improves work efficiency, and minimizes errors across various digital applications through contextual visual cues and interactive support.

---

## Table of Contents

- [Core Features](#core-features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Core Features

1. **Real-time UI Element Detection and Highlighting**: Analyzes screen content in real-time using computer vision and OCR to detect UI components, then highlights and outlines them with visual overlays to guide users toward target elements during task execution.
2. **Dual-mode Element Matching (Text and Image-based)**: Supports two complementary matching strategies: text-based matching using OCR-extracted content comparison, and image-based matching using CLIP vision-language embeddings for visual similarity analysis of UI components.
3. **Server-based Task Planning and Guidance**: Communicates with a backend server to receive intelligent task instructions, action plans, and step-by-step guidance based on screenshot analysis and conversation history, enabling adaptive task automation.
4. **Interactive Grid-based Overlay System**: Divides the screen into a 3x3 grid and displays an interactive overlay window that highlights detected UI elements with metadata (similarity scores, text content) to provide intuitive visual feedback for user guidance.
5. **Multi-stage UI Component Detection Pipeline**: Implements a comprehensive detection workflow combining text detection (OCR with Paddle/Google), non-text component detection using gradient maps and segmentation, and element merging to produce a unified representation of screen components.
6. **Screenshot Capture and Processing**: Captures full-screen screenshots and processes them through resizing, color space conversion, and coordinate transformation to prepare images for UI element detection and analysis.
7. **User Interaction Monitoring**: Monitors user actions on detected UI elements through grid-based coordinate tracking, confirming successful element interaction and triggering progression to the next task step.

---

## How It Works

Teept operates through a sophisticated multi-stage pipeline:

1. **Screen Analysis**: The system captures screenshots and analyzes their content using a combination of OCR and computer vision techniques.
2. **Element Detection**: Based on the [UIED](https://github.com/MulongXie/UIED) project, Teept detects UI elements through:
   - Text recognition using OCR (Paddle/Google)
   - Non-text component detection using gradient maps and segmentation
   - Element merging to create a unified interface representation
3. **Intelligent Identification**: The system uses GPT, CLIP neural networks, and text comparison to identify the required interface element from the detected components.
4. **Real-time Guidance**: Visual overlays highlight the identified elements, providing users with contextual step-by-step assistance.

This universal approach allows the system to be flexible, adaptable to different formats, and capable of working across various digital interfaces.

---

## Installation

Install teept-gui-assistant using one of the following methods:

**Build from source:**

1. Clone the teept-gui-assistant repository:
```sh
git clone https://github.com/andreygetmanov/teept-gui-assistant
```

2. Navigate to the project directory:
```sh
cd teept-gui-assistant
```

3. Install the project dependencies:
```sh
pip install -r requirements.txt
```

---

## Getting Started

1. **Create and activate a virtual environment:**

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

2. **Configure the environment:**

```bash
cp .env.example .env
```

Edit the `.env` file and set:

```bash
SITE_URL=jgsnapp.ru
```

Teept is currently designed to work through a server. Once configured, the assistant will analyze your screen in real-time, highlighting interface elements and providing contextual guidance to help you navigate digital interfaces more efficiently.

---

## Contributing

- **[Report Issues](https://github.com/andreygetmanov/teept-gui-assistant/issues)**: Submit bugs found or log feature requests for the project.

- **[Submit Pull Requests](https://github.com/andreygetmanov/teept-gui-assistant/tree/main/.github/CONTRIBUTING.md)**: To learn more about making a contribution to teept-gui-assistant.

---

## License

This project is protected under the Apache License 2.0. For more details, refer to the [LICENSE](https://github.com/andreygetmanov/teept-gui-assistant/tree/main/LICENSE) file.

---

## Citation

If you use this software, please cite it as below.

### APA format:

```
andreygetmanov (2026). teept-gui-assistant repository [Computer software]. https://github.com/andreygetmanov/teept-gui-assistant
```

### BibTeX format:

```bibtex
@misc{teept-gui-assistant,
    author = {andreygetmanov},
    title = {teept-gui-assistant repository},
    year = {2026},
    publisher = {github.com},
    journal = {github.com repository},
    howpublished = {\url{https://github.com/andreygetmanov/teept-gui-assistant.git}},
    url = {https://github.com/andreygetmanov/teept-gui-assistant.git}
}
```