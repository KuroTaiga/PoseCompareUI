# PoseCompareUI

A comprehensive UI tool for comparing pose estimation models, interpolation methods, and noise filtering techniques for human pose analysis.

## Overview

PoseCompareUI is a Gradio-based application designed to help researchers and developers compare different pose estimation approaches and post-processing techniques. The tool provides visual comparisons between various pose estimation models and filtering methods, making it easier to evaluate their performance on your own videos.

## Features

### Model Comparison
Compare multiple pose estimation models side by side:
- **MediaPipe**: Google's lightweight pose estimation framework
- **FourDHumans**: 3D human mesh reconstruction model
- **Sapiens**: Multiple variants (0.3b, 0.6b, 1b, 2b) of the Sapiens pose estimation model

### Noise Filtering Methods
Apply and compare different noise reduction filters:
- **Original**: No filtering applied
- **Butterworth**: Low-pass Butterworth filter for smooth trajectories
- **Chebyshev**: Chebyshev Type-I filter for noise reduction
- **Bessel**: Bessel filter with linear phase response

### Interpolation Techniques
Test and visualize various interpolation methods:
- **Kalman**: Predictive filtering using Kalman filter
- **Wiener**: Optimal filtering for stationary signals
- **Linear**: Simple linear interpolation
- **Bilinear**: Two-dimensional linear interpolation
- **Spline**: Smooth curve fitting using splines
- **Kriging**: Spatial interpolation method

### Rule Extraction
Extract movement rules and patterns from videos:
- Automatically identifies equipment used in the video
- Classifies movement patterns for different body parts (arms, legs, torso)
- Generates a comprehensive visualization with the original video, pose overlay, and detected features

## Getting Started

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)
- FFmpeg

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PoseCompareUI.git
cd PoseCompareUI
```

2. Run the setup script to create the conda environment and install dependencies:
```bash
bash setup.sh
```

3. Activate the environment:
```bash
conda activate posecompare
```

### Running the Application

Launch the application with:
```bash
python app.py
```

Then navigate to the provided URL (typically http://127.0.0.1:7860) in your web browser.

## Usage

1. **Upload or Record a Video**: Use the upload button or webcam option to provide a video for analysis.

2. **Model Comparison Tab**:
   - Select the models you want to compare
   - Choose a noise filtering method
   - Adjust the filter window size
   - Click "Process Video"

3. **Noise Tab**:
   - Select different noise filtering methods to compare
   - Adjust the filter window size
   - Click "Process Video"

4. **Interpolation Tab**:
   - Select interpolation methods to compare
   - Click "Process Video"

5. **Rule Extraction Tab**:
   - Click "Extract Rules" to analyze movement patterns
   - View the combined visualization showing original video, pose estimation, and detected features

## Project Structure

- `app.py`: Main application file with the Gradio interface
- `pose_processing.py`: Core pose processing functionality
- `pose_models.py`: Model wrappers for different pose estimation systems
- `sapiens_processor.py`: Implementation for the Sapiens model
- `RuleExtration/`: Module for extracting movement rules from videos
  - `real_time_debug.py`: Real-time analysis and visualization
  - `combine_video.py`: Utilities for combining video outputs
  - `src/`: Source files for rule extraction
    - `pose_estimation.py`: Pose detection and analysis
    - `feature_extraction.py`: Feature extraction from poses
    - `equipment_detection.py`: Equipment detection using YOLOv7

## Troubleshooting

- **CUDA/GPU Issues**: Make sure you have compatible NVIDIA drivers installed
- **Model Loading Errors**: Check that model weights are properly downloaded
- **Video Processing Failures**: Ensure FFmpeg is installed and accessible

## Acknowledgements

- MediaPipe from Google
- FourDHumans model
- Sapiens model
