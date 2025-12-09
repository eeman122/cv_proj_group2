"""
Incremental Structure from Motion Module
Extends two-view reconstruction to multiple views using PnP
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


class IncrementalSfM:
    """
    Manages the incremental SfM pipeline for multiple views.
    Uses PnP for pose estimation and triangulation for map expansion.
    """
    
    def __init__(self, K: np.ndarray):
        """
        Initialize incremental SfM system.
        
        Args:
            K: 3x3 camera intrinsic matrix
        """
        self.K = K
        self.map_points = []  # List of 3D points
        self.map_point_colors = []  # RGB colors for visualization
        self.camera_poses = []  # List of (R, t) tuples
        self.image_keypoints = []  # Keypoints for each image
        self.image_descriptors = []  # Descriptors for each image
        self.image_paths = []  # File paths for reference
        self.keypoint_to_mappoints = []  # For each image, maps keypoint index to list of map point indices
    
    def initialize_map(self, points_3d: np.ndarray, colors: np.ndarray,
                       R: np.ndarray, t: np.ndarray, kp1: List, des1: np.ndarray,
                       kp2: List, des2: np.ndarray,
                       match_indices: Optional[np.ndarray] = None) -> None:
        """
        Initialize the 3D map from two-view reconstruction.
        
        Args:
            points_3d: Nx3 array of initial 3D points
            colors: Nx3 array of RGB colors
            R: Rotation matrix of second camera
            t: Translation vector of second camera
            kp1: Keypoints from first image
            des1: Descriptors from first image
            kp2: Keypoints from second image
            des2: Descriptors from second image
            match_indices: Nx2 array of (kp1_idx, kp2_idx) for each 3D point
        """
        # Add first camera at origin
        self.camera_poses.append((np.eye(3), np.zeros((3, 1))))
        
        # Add second camera
        self.camera_poses.append((R, t))
        
        # Add 3D points to map
        for i, pt in enumerate(points_3d):
            self.map_points.append(pt)
            if i < len(colors):
                self.map_point_colors.append(colors[i])
            else:
                self.map_point_colors.append([0.5, 0.5, 0.5])
        
        # Create mapping for first and second images
        # match_indices[i] = (kp1_idx, kp2_idx) for 3D point i
        first_image_mapping = [[] for _ in range(len(kp1))]
        second_image_mapping = [[] for _ in range(len(kp2))]
        
        if match_indices is not None:
            # Each entry in match_indices gives us the keypoint indices for a 3D point
            for map_pt_idx, (kp1_idx, kp2_idx) in enumerate(match_indices):
                # map_pt_idx is the index into points_3d array
                # kp1_idx is the keypoint index in image 1
                # kp2_idx is the keypoint index in image 2
                if kp1_idx < len(first_image_mapping):
                    first_image_mapping[kp1_idx].append(map_pt_idx)
                if kp2_idx < len(second_image_mapping):
                    second_image_mapping[kp2_idx].append(map_pt_idx)
        
        self.keypoint_to_mappoints.append(first_image_mapping)
        self.keypoint_to_mappoints.append(second_image_mapping)
        
        self.image_keypoints.append(kp1)
        self.image_descriptors.append(des1)
        self.image_keypoints.append(kp2)
        self.image_descriptors.append(des2)
        
        print(f"✓ Map initialized with {len(self.map_points)} points and {len(self.camera_poses)} cameras")
        img1_with_3d = sum(1 for m in first_image_mapping if len(m) > 0)
        img2_with_3d = sum(1 for m in second_image_mapping if len(m) > 0)
        print(f"  Image 1: {len(kp1)} keypoints, {img1_with_3d} with 3D points")
        print(f"  Image 2: {len(kp2)} keypoints, {img2_with_3d} with 3D points")
    
    def solve_pnp(self, object_points: np.ndarray, image_points: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Solve PnP problem using RANSAC.
        
        Args:
            object_points: Nx3 array of 3D points
            image_points: Nx2 array of 2D image points
            
        Returns:
            R: Rotation matrix or None if failed
            t: Translation vector or None if failed
            inliers: Inlier mask
        """
        if len(object_points) < 4:
            print(f"Insufficient points for PnP: {len(object_points)}")
            return None, None, None
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points.astype(np.float32),
            image_points.astype(np.float32),
            self.K,
            None,
            iterationsCount=200,
            reprojectionError=8.0,
            confidence=0.99
        )
        
        if not success or rvec is None:
            print("PnP failed to find a solution")
            return None, None, None
        
        R, _ = cv2.Rodrigues(rvec)
        t = tvec
        
        return R, t, inliers
    
    def triangulate_new_points(self, kp_prev: List, des_prev: np.ndarray,
                              kp_new: List, des_new: np.ndarray,
                              R_prev: np.ndarray, t_prev: np.ndarray,
                              R_new: np.ndarray, t_new: np.ndarray) -> Tuple[List, List]:
        """
        Find new 3D points by matching features and triangulating.
        
        Args:
            kp_prev, des_prev: Previous image keypoints and descriptors
            kp_new, des_new: New image keypoints and descriptors
            R_prev, t_prev: Previous camera pose
            R_new, t_new: New camera pose
            
        Returns:
            new_points: List of new 3D points
            new_colors: List of colors for new points
        """
        # Match features
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = matcher.knnMatch(des_prev, des_new, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 4:
            print(f"Insufficient feature matches for triangulation: {len(good_matches)}")
            return [], []
        
        # Extract matching points
        pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in good_matches])
        pts_new = np.float32([kp_new[m.trainIdx].pt for m in good_matches])
        
        # Construct projection matrices
        P_prev = self.K @ np.hstack((R_prev, t_prev))
        P_new = self.K @ np.hstack((R_new, t_new))
        
        # Normalize points
        pts_prev_norm = cv2.undistortPoints(pts_prev.reshape(-1, 1, 2), self.K, None)
        pts_new_norm = cv2.undistortPoints(pts_new.reshape(-1, 1, 2), self.K, None)
        
        # Triangulate
        points_4d = cv2.triangulatePoints(P_prev, P_new, pts_prev_norm, pts_new_norm)
        points_3d = (points_4d[:3] / points_4d[3]).T
        
        # Filter valid points (positive depth in both cameras)
        new_points = []
        new_colors = []
        
        for i, pt in enumerate(points_3d):
            # Check depth in camera 1
            if pt[2] <= 0:
                continue
            
            # Check depth in camera 2
            pt_cam2 = R_new @ pt.reshape(3, 1) + t_new
            if pt_cam2[2, 0] <= 0:
                continue
            
            new_points.append(pt)
            # Color is fixed for this implementation
            new_colors.append([0.5, 0.5, 0.5])
        
        return new_points, new_colors
    
    def add_image(self, kp: List, des: np.ndarray, img_rgb: Optional[np.ndarray] = None) -> bool:
        """
        Add a new image to the reconstruction.
        
        Args:
            kp: Keypoints in the new image
            des: Descriptors for the new image
            img_rgb: RGB image for color extraction (optional)
            
        Returns:
            success: Whether the image was successfully added
        """
        if len(self.map_points) == 0:
            print("ERROR: Map is empty. Initialize with two-view reconstruction first.")
            return False
        
        if des is None:
            print("ERROR: Descriptors are None")
            return False
        
        # Try matching against all previous images to find 2D-3D correspondences
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        object_points = []
        image_points = []
        
        # Iterate through all previous images starting from the most recent
        for img_idx in range(len(self.image_descriptors) - 1, -1, -1):
            prev_des = self.image_descriptors[img_idx]
            prev_kp = self.image_keypoints[img_idx]
            prev_mapping = self.keypoint_to_mappoints[img_idx]
            
            if prev_des is None:
                continue
            
            # Match with this previous image
            matches = matcher.knnMatch(prev_des, des, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            # DEBUG
            valid_3d_in_prev = sum(1 for i, m in enumerate(prev_mapping) if len(m) > 0)
            print(f"  Checking image {img_idx+1}: {len(good_matches)} matches, {valid_3d_in_prev} kpts with 3D")
            
            # Build 2D-3D correspondences from these matches
            for m in good_matches:
                prev_idx = m.queryIdx
                curr_idx = m.trainIdx
                
                # Check if this keypoint has corresponding map points
                if prev_idx < len(prev_mapping) and len(prev_mapping[prev_idx]) > 0:
                    # Use all available map points for this keypoint
                    for map_pt_idx in prev_mapping[prev_idx]:
                        if map_pt_idx < len(self.map_points):
                            object_points.append(self.map_points[map_pt_idx])
                            image_points.append(kp[curr_idx].pt)
            
            # Stop if we have enough correspondences
            if len(object_points) >= 4:
                print(f"Found {len(object_points)} 2D-3D correspondences from image {img_idx+1}")
                break
        
        if len(object_points) < 4:
            print("ERROR: Not enough 2D-3D correspondences")
            return False
        
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)
        
        # Solve PnP
        print(f"Solving PnP with {len(object_points)} correspondences...")
        R_new, t_new, inliers = self.solve_pnp(object_points, image_points)
        
        if R_new is None:
            print("PnP pose estimation failed")
            return False
        
        print(f"✓ PnP pose estimated with {len(inliers) if inliers is not None else 0} inliers")
        
        # Add to camera poses
        self.camera_poses.append((R_new, t_new))
        self.image_keypoints.append(kp)
        self.image_descriptors.append(des)
        
        # Create mapping for this new image (empty lists means no map point assigned)
        new_mapping = [[] for _ in range(len(kp))]
        self.keypoint_to_mappoints.append(new_mapping)
        
        # Expand map with new triangulated points
        R_prev, t_prev = self.camera_poses[-2]
        prev_kp = self.image_keypoints[-2]
        prev_des = self.image_descriptors[-2]
        
        new_points, new_colors = self.triangulate_new_points(
            prev_kp, prev_des, kp, des,
            R_prev, t_prev, R_new, t_new
        )
        
        for pt, color in zip(new_points, new_colors):
            self.map_points.append(pt)
            self.map_point_colors.append(color)
        
        print(f"Added {len(new_points)} new 3D points")
        print(f"Total points in map: {len(self.map_points)}")
        print(f"Total cameras: {len(self.camera_poses)}")
        
        return True
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the current reconstruction.
        
        Returns:
            stats: Dictionary with reconstruction statistics
        """
        if len(self.map_points) == 0:
            return {'num_points': 0, 'num_cameras': 0}
        
        points_array = np.array(self.map_points)
        
        stats = {
            'num_points': len(self.map_points),
            'num_cameras': len(self.camera_poses),
            'x_range': (points_array[:, 0].min(), points_array[:, 0].max()),
            'y_range': (points_array[:, 1].min(), points_array[:, 1].max()),
            'z_range': (points_array[:, 2].min(), points_array[:, 2].max()),
            'mean_depth': float(np.mean(points_array[:, 2]))
        }
        
        return stats
    
    def export_point_cloud(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Export the current point cloud.
        
        Returns:
            points: Nx3 array of 3D points
            colors: Nx3 array of RGB colors
        """
        return np.array(self.map_points), np.array(self.map_point_colors)
