# CV Project - Structure from Motion (Group 2)

A complete Structure from Motion (SfM) implementation that reconstructs 3D scenes from multiple 2D images. This project includes feature detection, matching, camera pose estimation, and 3D point cloud generation with bundle adjustment.

## 📁 Project Structure

```
├── sfm.py                      # Main SfM reconstruction pipeline
├── run_sfm.py                  # Quick-start script to run the pipeline
├── feature_matcher.py          # Feature detection and matching (SIFT/SURF)
├── camera_calibration.py       # Camera intrinsic parameter calibration
├── two_view_geometry.py        # Essential/Fundamental matrix computation
├── bundle_adjustment.py        # Optimization of camera poses and 3D points
├── incremental_sfm.py          # Incremental reconstruction approach
├── match.py                    # Utility functions for feature matching
├── utils.py                    # Common utilities and helper functions
├── cv.ipynb                    # Jupyter notebook for interactive exploration
├── cv_dev3.ipynb              # Development notebook with experiments
├── sfm_interactive.ipynb       # Interactive SfM pipeline notebook
├── Features/                   # Detected features for images
├── Depth Maps/                 # Computed depth maps
├── Point Clouds/               # Generated 3D point cloud files
├── Three.js Tour/              # 3D visualization for web viewing
└── README.md                   # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- OpenCV
- NumPy, SciPy
- Open3D (optional, for visualization)

### Installation

```bash
pip install -r requirements.txt
```

### Quick Start

Run the complete SfM pipeline on your images:

```bash
python run_sfm.py --source_dir /path/to/images --root_dir /path/to/output
```

**Options:**
- `--source_dir`: Path to your images folder
- `--root_dir`: Output directory for SfM results
- `--num_images`: Limit the number of images (default: all)
- `--feat_type`: Feature type to use - 'sift' or 'surf' (default: sift)

## 🔧 Core Components

### 1. Feature Detection & Matching (`feature_matcher.py`)
- Detects features using SIFT or SURF algorithms
- Matches features between image pairs
- Filters matches using RANSAC for robustness

### 2. Camera Calibration (`camera_calibration.py`)
- Computes camera intrinsic parameters
- Calibrates lens distortion
- Generates intrinsic matrix K

### 3. Two-View Geometry (`two_view_geometry.py`)
- Computes Essential Matrix using RANSAC
- Calculates camera pose from Essential Matrix
- Triangulates 3D points from feature correspondences

### 4. Main Pipeline (`sfm.py`)
- Orchestrates the incremental reconstruction
- Manages 3D point tracking
- Handles new view registration
- Accumulates 3D point cloud

### 5. Bundle Adjustment (`bundle_adjustment.py`)
- Optimizes camera poses and 3D point positions
- Minimizes reprojection error
- Refines the reconstruction

### 6. Incremental Reconstruction (`incremental_sfm.py`)
- Implements incremental SfM approach
- Adds images one at a time
- Maintains consistency across views

## 📓 Jupyter Notebooks

- **sfm_interactive.ipynb**: Step-by-step interactive pipeline execution
- **cv.ipynb**: Exploration and visualization of results
- **cv_dev3.ipynb**: Development and experimental features

## 📊 Output Files

- **Features/**: Detected feature keypoints and descriptors
- **Point Clouds/**: 3D point cloud files (PLY, OBJ formats)
- **Depth Maps/**: Depth maps for each view
- **Three.js Tour/**: Web-based 3D visualization

## 🎯 Workflow

1. **Feature Detection**: Extract SIFT/SURF features from all images
2. **Feature Matching**: Match features between image pairs
3. **Initialization**: Build initial reconstruction from first two views
4. **Incremental Addition**: Sequentially add remaining images
5. **Bundle Adjustment**: Optimize poses and 3D points
6. **Visualization**: Generate point clouds and 3D tours

## 📈 Performance Considerations

- Feature matching can be slow for large image sets
- Use `--num_images` to limit the number of images for testing
- SIFT is slower but more accurate; SURF is faster
- Bundle adjustment time increases with more images

## 🔍 Troubleshooting

**Issue**: Open3D not available
- Solution: `pip install open3d`

**Issue**: No matches found between images
- Check image overlap - ensure images have sufficient overlap
- Try adjusting matching thresholds in `feature_matcher.py`

**Issue**: Poor reconstruction quality
- Verify camera calibration parameters
- Check image resolution and overlap
- Increase number of images for better coverage

## 📝 Notes

- Requires images with significant overlap for good reconstruction
- Camera calibration should be done beforehand for best results
- Results improve with more images and better baseline distances

## 👥 Contributors

Group 2 - Eeman Adnan and Kashaf Gohar
