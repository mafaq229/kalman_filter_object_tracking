# Kalman Filter Object Tracking

This project implements object tracking using Kalman filters, based on the CS6476 Computer Vision course assignment. The implementation provides robust tracking capabilities for various objects in video sequences.

## Features

- Kalman Filter implementation for object tracking
- Support for both position and velocity state estimation
- Configurable process and measurement noise parameters
- Easy-to-use API for tracking objects in video sequences

## Installation

This project uses `uv` as the package manager. To install dependencies:

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync
```

## Project Structure

```
.
├── src/
│   └── kalman.py      # Kalman Filter implementation
├── main.py            # 
└── README.md          # This file
```

## Dependencies

- numpy
- opencv-python


