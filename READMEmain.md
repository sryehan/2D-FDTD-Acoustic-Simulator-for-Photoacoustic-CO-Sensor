# 2D FDTD Acoustic Simulator for Photoacoustic CO₂ Sensor by python 

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

A high-performance 2D Finite-Difference Time-Domain (FDTD) simulator for photoacoustic gas sensors, specifically optimized for CO₂ detection in resonant photoacoustic cells.

## 📋 Overview

This simulator models acoustic wave propagation in a resonant photoacoustic cell with the following geometry:
- **Pipe**: 7 mm diameter × 57.17 mm length
- **Buffers**: 36 mm diameter × 28.58 mm length (both sides)

The simulation uses 2D FDTD methods to solve the linearized acoustic equations, providing insights into:
- Pressure field distribution
- Resonance frequencies
- Q-factors
- Signal amplification with wavelength modulation spectroscopy (WMS)

## 🚀 Features

- **2D FDTD solver** with optimized performance
- **Wavelength Modulation Spectroscopy (WMS)** simulation (f_mod and 2f)
- **Configurable geometry** for different resonator designs
- **Real-time visualization** of pressure fields
- **Frequency domain analysis** with FFT
- **Q-factor calculation** from resonance peaks
- **Modular design** for easy extension

## 📊 Results Example

![Pressure Field](docs/images/pressure_field.png)
*Pressure distribution in the resonator at resonance*

![Frequency Spectrum](docs/images/spectrum.png)
*Frequency spectrum showing fundamental and 2f components*

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install from source
```bash
git clone https://github.com/yourusername/pas-fdtd-simulator.git
cd pas-fdtd-simulator
pip install -r requirements.txt
pip install -e .
