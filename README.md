# SignalFlow Lab

A Digital Signal Processing (DSP) toolkit in Python with a Tkinter GUI for signal generation, processing, filtering, and visualization.

## Overview

SignalFlow Lab provides an interactive desktop application for performing common DSP operations. It covers signal generation, basic operations, digitization, FIR filter design with windowing, and result visualization — all through a graphical interface.

## Features

- **Signal Generation** — generate standard waveforms
- **Basic Operations** — arithmetic and manipulation on signals
- **Signal Digitization** — convert continuous signals to discrete
- **FIR Filter Design** — low-pass, high-pass, band-pass, and band-stop filters using windowing methods (Rectangular, Hanning, Hamming, Blackman)
- **Automatic window selection** based on stopband attenuation requirements
- **Visualization** — plot signals and results with matplotlib
- **Test suite** — per-task test scripts to validate output

## Project Structure

```
├── main.py                 # Entry point — launches the Tkinter GUI
├── Filteration.py          # FIR filter design with windowing
├── gui/
│   └── app.py              # Main GUI application (SignalProcessorApp)
├── processing/
│   ├── basic_ops.py        # Basic signal operations
│   ├── generation.py       # Signal generation
│   └── signal_digitize.py  # Signal digitization
├── visualization/          # Plotting utilities
├── results/                # Saved output results
├── tests/                  # Task-based test scripts
├── Testing/                # Additional test cases
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

Python >= 3.10 is required. A virtual environment is recommended.

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the app

```bash
python main.py
```

> On Windows: `python main.py`

## Running Tests

From the project root, run the test script for the task you want to verify:

```bash
python tests/task1/*.py
```

## Dependencies

- `numpy`
- `matplotlib`
- `pandas`
- `pillow`
- `scipy` (used in filter design)
- `tkinter` (built into Python)

## License

MIT
