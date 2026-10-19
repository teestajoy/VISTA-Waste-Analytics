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

Dependencies have not yet been finalized or installed as part of the initial setup.

## Current Status

The project is in its initial setup stage. The source package structure and empty Python package marker files have been created. The computer-vision pipeline, model integration, assessments, impact calculations, API, and dashboard still need to be implemented and tested.

## Responsible Estimates

Environmental and economic outputs should document their sources, units, assumptions, and limitations. Results based only on images should be presented as estimates rather than precise measurements.

