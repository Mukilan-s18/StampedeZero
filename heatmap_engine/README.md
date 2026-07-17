# Heatmap Engine (Engineer 2)

This directory contains the crowd density estimation code that bypasses YOLO and uses a CNN (CSRNet) to analyze pixel density in massively crowded stadiums.

## Setup

1. **Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Mac/Linux
   source venv/bin/activate
   ```

2. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure you have the correct version of PyTorch for your system (CPU or CUDA). Check [PyTorch's website](https://pytorch.org/) for the command if `pip install torch` fails to detect your GPU.*

3. **Get the Weights**:
   Download a pre-trained `model_best.pth.tar` for CSRNet (trained on ShanghaiTech Part A) and place it in this directory.
   - You can find them in GitHub repositories like [leeyeehoo/CSRNet-pytorch](https://github.com/leeyeehoo/CSRNet-pytorch) under their Releases or linked Baidu/Google Drive links.

## Usage for Engineer 4 (UI Handoff)

The `dense_crowd_ai.py` file exposes a `DensityEstimator` class that makes inference incredibly easy.

```python
import cv2
from dense_crowd_ai import DensityEstimator

# 1. Initialize once with the weights file
estimator = DensityEstimator('model_best.pth.tar')

# 2. Read a frame from video or image
frame = cv2.imread('test_crowd.jpg')

# 3. Process the frame
result = estimator.process_frame(frame)

# 4. Extract data
count = result['estimated_count']
heatmap_frame = result['heatmap_frame']

print(f"Estimated Crowd Count: {count}")
cv2.imwrite('output_heatmap.jpg', heatmap_frame)
```
