"""
Feature Detection and Matching Module
Implements SIFT/ORB feature detection and matching with Lowe's ratio test
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


class FeatureMatcher:
    """
    Detects and matches features between image pairs using SIFT or ORB.
    Applies Lowe's ratio test for quality filtering.
    """
    
    def __init__(self, detector_type='SIFT', matcher_type='BF'):
        """
        Initialize feature matcher.
        
        Args:
            detector_type: 'SIFT', 'ORB', or 'AKAZE'
            matcher_type: 'BF' (Brute Force) or 'FLANN'
        """
        self.detector_type = detector_type
        self.matcher_type = matcher_type
        
        if detector_type == 'SIFT':
            self.detector = cv2.SIFT_create()
        elif detector_type == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=2000)
        elif detector_type == 'AKAZE':
            self.detector = cv2.AKAZE_create()
        else:
            raise ValueError(f"Unknown detector: {detector_type}")
    
    def detect_and_compute(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Detect keypoints and compute descriptors.
        
        Args:
            image: Input image (preferably grayscale)
            
        Returns:
            keypoints: List of cv2.KeyPoint objects
            descriptors: numpy array of shape (N, descriptor_dim)
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray, 
                      ratio_threshold: float = 0.7) -> List:
        """
        Match descriptors between two images using Lowe's ratio test.
        
        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image
            ratio_threshold: Lowe's ratio test threshold
            
        Returns:
            good_matches: List of cv2.DMatch objects passing the ratio test
        """
        if self.matcher_type == 'BF':
            if self.detector_type == 'SIFT':
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            else:
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            if self.detector_type == 'SIFT':
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
            else:
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm=FLANN_INDEX_LSH,
                                  table_number=6, key_size=12,
                                  multi_probe_level=1)
                search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = matcher.knnMatch(desc1, desc2, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def visualize_matches(self, img1: np.ndarray, img2: np.ndarray,
                         kp1: List, kp2: List, matches: List,
                         max_matches: int = 50) -> np.ndarray:
        """
        Visualize matched features between two images.
        
        Args:
            img1, img2: Input images
            kp1, kp2: Keypoints from respective images
            matches: List of cv2.DMatch objects
            max_matches: Maximum number of matches to display
            
        Returns:
            img_matches: Image showing matched features
        """
        # Convert to RGB if needed
        if len(img1.shape) == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if len(img2.shape) == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        if img1.shape[2] == 3 and img1.dtype != np.uint8:
            img1 = (img1 * 255).astype(np.uint8)
        if img2.shape[2] == 3 and img2.dtype != np.uint8:
            img2 = (img2 * 255).astype(np.uint8)
        
        # Draw matches
        draw_params = dict(matchColor=(0, 255, 0),
                          singlePointColor=(255, 0, 0),
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2,
                                     matches[:max_matches], None, **draw_params)
        return img_matches
