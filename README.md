# VISTA — Vision-based Intelligent System for Waste Triage and Analytics

## Project Overview

VISTA is a planned computer-vision system for analyzing municipal solid waste and electronic waste. Its intended pipeline will detect waste items in images, classify them, assess contamination and physical condition, and estimate environmental impact and recoverable economic value.

These estimates will depend on documented reference factors and assumptions. Image-only estimates may be uncertain, particularly when an item's mass or material composition is unknown.

## Planned Features

- Waste object detection.
- Separate classification categories for municipal solid waste and electronic waste.
- Contamination and physical-condition assessment.
- Estimated CO2-equivalent savings and recoverable economic value.
- Visual explanations of model predictions.
- An API and an interactive dashboard.
- Tests for the implemented modules.

These are development goals, not claims of completed functionality.

## Technology Stack

The planned stack includes Python, PyTorch, YOLO, OpenCV or Pillow, FastAPI, Streamlit, and Grad-CAM tooling. Specific dependencies and versions will be selected and tested during setup.

## Project Structure

- `src/vista/detection/` — waste object detection.
- `src/vista/classification/` — waste category classification.
- `src/vista/assessment/` — contamination and physical-condition assessment.
- `src/vista/impact/` — environmental and economic estimates.
- `src/vista/explainability/` — visual explanations of predictions.
- `src/vista/api/` — API functionality.
- `src/vista/common/` — shared utilities and settings.
- `data/` — datasets and reference data.
- `models/` — model artifacts.
- `dashboard/` — dashboard application.
- `hardware/` — hardware and deployment-related work.
- `notebooks/` — experiments and learning exercises.
- `tests/` — automated tests.
- `docs/` — architecture and reference documentation.

## Development Setup

This project targets Python 3.12.

On Windows PowerShell, activate the project virtual environment from the repository root:

`.\venv\Scripts\Activate.ps1`

Verify the active Python version:

`python --version`

The `ultralytics` requirement is listed in `requirements.txt`. Set up its PyTorch dependencies in this order:

1. Use the official PyTorch installation selector to choose and install a compatible PyTorch and torchvision build for this Windows machine, based on whether CPU or CUDA will be used.
2. Record the selected PyTorch and torchvision versions in a pip constraints file, then install the remaining project dependencies from `requirements.txt` using that constraints file. Review the dependency resolution plan first, such as with pip's `--dry-run` option. The constraints should keep pip from silently replacing the selected build. If resolution proposes replacing or changing PyTorch or torchvision, stop and review the installation plan before proceeding.

## YOLO Smoke Test

After installing the project requirements and the selected PyTorch/torchvision build, run this command from the repository root with a local image path:

`.\venv\Scripts\python.exe scripts\smoke_yolo.py "C:\path\to\your\image.jpg"`

The first run may download the `yolo26n.pt` model weights. The pretrained model is a general-purpose detector, not a VISTA waste-specific model; its labels do not determine an item's material, recyclability, contamination, or condition.

## Current Status

The image validation component and YOLO detector interface have been implemented. Real YOLO inference has not yet been verified. Waste-specific classification, contamination and condition assessment, explainability, impact estimation, and the dashboard and API have not yet been verified as implemented.

## Responsible Estimates

Environmental and economic outputs should document their sources, units, assumptions, and limitations. Results based only on images should be presented as estimates rather than precise measurements.

