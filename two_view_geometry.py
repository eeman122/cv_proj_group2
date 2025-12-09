"""
Two-View Geometry Module
Implements essential matrix estimation, pose recovery, and triangulation
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional


class TwoViewGeometry:
    """
    Handles all two-view geometric computations for SfM:
    - Essential matrix estimation with RANSAC
    - Camera pose recovery
    - 3D point triangulation
    - Cheirality check for pose disambiguation
    """
    
    def __init__(self, K: np.ndarray):
        """
        Initialize with camera intrinsic matrix.
        
        Args:
            K: 3x3 camera intrinsic matrix
        """
        self.K = K
        self.f_x = K[0, 0]
        self.f_y = K[1, 1]
        self.c_x = K[0, 2]
        self.c_y = K[1, 2]
    
    def estimate_essential_matrix(self, pts1: np.ndarray, pts2: np.ndarray,
                                  confidence: float = 0.9999,
                                  threshold: float = 1.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimate essential matrix using RANSAC.
        
        Args:
            pts1, pts2: Matched point coordinates (N x 2)
            confidence: RANSAC confidence level
            threshold: RANSAC reprojection error threshold
            
        Returns:
            E: 3x3 essential matrix
            mask: Inlier mask
        """
        if len(pts1) < 8:
            print(f"Warning: Only {len(pts1)} points, need at least 8")
            return None, None
        
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K,
                                       method=cv2.RANSAC,
                                       prob=confidence,
                                       threshold=threshold)
        
        if E is None:
            print("Essential matrix estimation failed")
            return None, None
        
        return E, mask
    
    def recover_pose(self, E: np.ndarray, pts1: np.ndarray, 
                    pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Recover camera pose from essential matrix.
        
        Args:
            E: Essential matrix
            pts1, pts2: Inlier matched points
            
        Returns:
            R: Rotation matrix (3x3)
            t: Translation vector (3x1)
            mask: Inlier mask from pose recovery
        """
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)
        return R, t, mask
    
    def triangulate_points(self, pts1: np.ndarray, pts2: np.ndarray,
                          R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Triangulate 3D points from two views.
        
        Args:
            pts1, pts2: Matched normalized image coordinates
            R: Relative rotation matrix
            t: Relative translation vector
            
        Returns:
            points_3d: Nx3 array of triangulated 3D points
        """
        # Construct projection matrices
        P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.K @ np.hstack((R, t))
        
        # Normalize image coordinates
        pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), self.K, None)
        pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), self.K, None)
        
        # Triangulate
        points_4d = cv2.triangulatePoints(P1, P2, pts1_norm, pts2_norm)
        
        # Convert from homogeneous to 3D coordinates
        points_3d = points_4d[:3] / points_4d[3]
        points_3d = points_3d.T
        
        return points_3d
    
    def cheirality_check(self, points_3d: np.ndarray, R: np.ndarray,
                        t: np.ndarray) -> np.ndarray:
        """
        Check if 3D points are in front of both cameras (positive depth).
        
        Args:
            points_3d: Nx3 array of 3D points
            R: Rotation matrix
            t: Translation vector
            
        Returns:
            valid_mask: Boolean mask of valid points
        """
        # Points in camera 1 frame (identity)
        depth_cam1 = points_3d[:, 2]
        valid_cam1 = depth_cam1 > 0
        
        # Points in camera 2 frame
        points_cam2 = (R @ points_3d.T + t).T
        depth_cam2 = points_cam2[:, 2]
        valid_cam2 = depth_cam2 > 0
        
        return valid_cam1 & valid_cam2
    
    def disambiguate_pose(self, pts1: np.ndarray, pts2: np.ndarray,
                         E: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Find the correct pose among the four possible solutions.
        
        Args:
            pts1, pts2: Matched points
            E: Essential matrix
            
        Returns:
            R: Correct rotation matrix
            t: Correct translation vector
            points_3d: Triangulated 3D points for the correct pose
            valid_mask: Boolean mask indicating which input points produced valid 3D points
        """
        # Get four possible solutions from E
        U, _, Vt = np.linalg.svd(E)
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        
        solutions = [
            (U @ W @ Vt, U[:, 2]),
            (U @ W @ Vt, -U[:, 2]),
            (U @ W.T @ Vt, U[:, 2]),
            (U @ W.T @ Vt, -U[:, 2])
        ]
        
        best_solution = None
        max_valid_points = 0
        
        for R_candidate, t_candidate in solutions:
            # Ensure R is a proper rotation (det = 1)
            if np.linalg.det(R_candidate) < 0:
                R_candidate = -R_candidate
            
            t_candidate = t_candidate.reshape(3, 1)
            
            # Triangulate
            points_3d = self.triangulate_points(pts1, pts2, R_candidate, t_candidate)
            
            # Check cheirality
            valid_mask = self.cheirality_check(points_3d, R_candidate, t_candidate)
            num_valid = np.sum(valid_mask)
            
            if num_valid > max_valid_points:
                max_valid_points = num_valid
                best_solution = (R_candidate, t_candidate, points_3d[valid_mask], valid_mask)
        
        if best_solution is None:
            print("Warning: No valid pose solution found")
            R, t = np.eye(3), np.zeros((3, 1))
            points_3d = np.array([])
            valid_mask = np.array([], dtype=bool)
        else:
            R, t, points_3d, valid_mask = best_solution
        
        return R, t, points_3d, valid_mask
    
    def two_view_reconstruction(self, img1_path: str, img2_path: str,
                               kp1: List, kp2: List, matches: List) -> Dict:
        """
        Complete two-view reconstruction pipeline.
        
        Args:
            img1_path, img2_path: Image file paths
            kp1, kp2: Keypoints from respective images
            matches: Matched features
            
        Returns:
            results: Dictionary with reconstruction results
        """
        if len(matches) < 8:
            print(f"Insufficient matches: {len(matches)}")
            return None
        
        # Extract matched point coordinates
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        match_indices = np.array([(m.queryIdx, m.trainIdx) for m in matches])
        
        # Estimate essential matrix
        E, mask = self.estimate_essential_matrix(pts1, pts2)
        if E is None:
            return None
        
        inlier_pts1 = pts1[mask.ravel() == 1]
        inlier_pts2 = pts2[mask.ravel() == 1]
        inlier_match_indices = match_indices[mask.ravel() == 1]
        
        print(f"Essential matrix inliers: {len(inlier_pts1)}")
        
        # Recover pose with cheirality check
        R, t, points_3d, valid_mask = self.disambiguate_pose(inlier_pts1, inlier_pts2, E)
        
        # Filter match indices to only include valid triangulated points
        inlier_match_indices = inlier_match_indices[valid_mask]
        
        print(f"3D points after cheirality check: {len(points_3d)}")
        
        return {
            'E': E,
            'R': R,
            't': t,
            'points_3d': points_3d,
            'inlier_pts1': inlier_pts1[valid_mask],
            'inlier_pts2': inlier_pts2[valid_mask],
            'inlier_match_indices': inlier_match_indices,
            'num_inliers': len(points_3d)
        }
