"""
Camera Calibration Module
Estimates camera intrinsic matrix from image properties or EXIF data
"""

import cv2
import numpy as np
from PIL import Image
import PIL.ExifTags


class CameraCalibration:
    """
    Handles camera intrinsic matrix estimation.
    """
    
    @staticmethod
    def extract_exif_data(image_path: str) -> dict:
        """
        Extract EXIF metadata from image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            exif_data: Dictionary of EXIF tags and values
        """
        exif_data = {}
        try:
            img = Image.open(image_path)
            if hasattr(img, '_getexif') and img._getexif() is not None:
                for tag_id, value in img._getexif().items():
                    tag = PIL.ExifTags.TAGS.get(tag_id, tag_id)
                    exif_data[tag] = value
        except Exception as e:
            print(f"Warning: Could not read EXIF data: {e}")
        
        return exif_data
    
    @staticmethod
    def estimate_intrinsics_from_image(image_path: str, 
                                      focal_length_mm: float = 50.0,
                                      sensor_width_mm: float = 36.0) -> np.ndarray:
        """
        Estimate camera intrinsic matrix from image properties.
        
        Args:
            image_path: Path to image file
            focal_length_mm: Focal length in mm (default: 50mm standard)
            sensor_width_mm: Sensor width in mm (default: 36mm for full frame)
            
        Returns:
            K: 3x3 camera intrinsic matrix
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        h, w = img.shape[:2]
        
        # Convert focal length from mm to pixels
        focal_length_pixels = (focal_length_mm / sensor_width_mm) * w
        
        # Principal point at image center
        cx = w / 2
        cy = h / 2
        
        K = np.array([
            [focal_length_pixels, 0, cx],
            [0, focal_length_pixels, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        print(f"Estimated intrinsic matrix from {image_path}:")
        print(f"  Image size: {w}x{h}")
        print(f"  Focal length: {focal_length_pixels:.1f} pixels")
        print(f"  Principal point: ({cx:.1f}, {cy:.1f})")
        
        return K
    
    @staticmethod
    def estimate_intrinsics_from_exif(image_path: str) -> np.ndarray:
        """
        Estimate camera intrinsic matrix using EXIF focal length.
        
        Args:
            image_path: Path to image file
            
        Returns:
            K: 3x3 camera intrinsic matrix
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        h, w = img.shape[:2]
        
        exif_data = CameraCalibration.extract_exif_data(image_path)
        
        focal_length_pixels = w  # Default estimate
        
        if 'FocalLengthIn35mmFilm' in exif_data:
            focal_35mm = exif_data['FocalLengthIn35mmFilm']
            sensor_width_35mm = 36.0
            sensor_width_actual = 7.0
            focal_actual = (focal_35mm * sensor_width_actual) / sensor_width_35mm
            focal_length_pixels = (focal_actual * w) / sensor_width_actual
            print(f"Using 35mm equivalent focal length: {focal_35mm}mm")
        elif 'FocalLength' in exif_data:
            focal_length = exif_data['FocalLength']
            if isinstance(focal_length, tuple):
                focal_length = focal_length[0] / focal_length[1]
            focal_length_pixels = (focal_length * w) / 7.0
            print(f"Using EXIF focal length: {focal_length}mm")
        else:
            print("Warning: No focal length in EXIF, using default estimate")
        
        cx = w / 2
        cy = h / 2
        
        K = np.array([
            [focal_length_pixels, 0, cx],
            [0, focal_length_pixels, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return K
    
    @staticmethod
    def calibrate_from_image_sequence(image_paths: list) -> np.ndarray:
        """
        Estimate intrinsic matrix from multiple images and average.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            K: 3x3 camera intrinsic matrix (averaged)
        """
        K_matrices = []
        
        for path in image_paths:
            try:
                K = CameraCalibration.estimate_intrinsics_from_exif(path)
                K_matrices.append(K)
            except Exception as e:
                print(f"Warning: Could not calibrate from {path}: {e}")
        
        if len(K_matrices) == 0:
            # Fallback to default
            print("Warning: Using default intrinsic matrix")
            w, h = 1920, 1440
            focal_length = w
            K = np.array([
                [focal_length, 0, w/2],
                [0, focal_length, h/2],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            K = np.mean(K_matrices, axis=0)
            print(f"\n✓ Averaged intrinsic matrix from {len(K_matrices)} images")
        
        return K
