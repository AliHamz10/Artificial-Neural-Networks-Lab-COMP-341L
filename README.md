# Artificial Neural Networks Lab - COMP-341L

This repository contains lab reports and manuals for the Artificial Neural Networks course (COMP-341L).

## Structure

- **Lab Reports/**: Contains individual lab report folders (Lab 1 through Lab 12)
- **Lab Manuals/**: Contains lab manual documents and resources
- **Lab 1/**, **Lab 2/**, etc.: Contains lab implementation files and reports

## Virtual Environment Setup

This project uses a single virtual environment for all labs.

### Initial Setup

1. Create virtual environment:
   ```bash
   python3 -m venv venv
   ```

2. Activate virtual environment:
   ```bash
   source venv/bin/activate  # macOS/Linux
   # or
   venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Using the Virtual Environment

- **Activate**: `source venv/bin/activate` (macOS/Linux) or `venv\Scripts\activate` (Windows)
- **Deactivate**: `deactivate`
- **Check if active**: Look for `(venv)` prefix in your terminal prompt

## Requirements

- Python 3.8 or higher
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0

## Usage

Place your lab reports in the respective `Lab Reports/Lab X/` folders and lab manuals in the `Lab Manuals/` folder.
