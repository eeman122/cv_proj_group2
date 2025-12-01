import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


class IncrementalSfM:
 
    def __init__(self, K: np.ndarray):

        self.K = K
        self.map_points = []  # List of 3D points
        self.map_point_colors = []  # RGB colors for visualization
        self.camera_poses = []  # List of (R, t) tuples
        self.image_keypoints = []  # Keypoints for each image
        self.image_descriptors = []  # Descriptors for each image
        self.image_paths = []  # File paths for reference
    
    def initialize_map(self, points_3d: np.ndarray, colors: np.ndarray,
                       R: np.ndarray, t: np.ndarray, kp: List, des: np.ndarray) -> None:

        self.camera_poses.append((np.eye(3), np.zeros((3, 1))))        
        self.camera_poses.append((R, t))
        
        for i, pt in enumerate(points_3d):
            self.map_points.append(pt)
            if i < len(colors):
                self.map_point_colors.append(colors[i])
            else:
                self.map_point_colors.append([0.5, 0.5, 0.5])
        
        self.image_keypoints.append(kp)
        self.image_descriptors.append(des)
        
        print(f"Map initialized with {len(self.map_points)} points and {len(self.camera_poses)} cameras")
    
    def solve_pnp(self, object_points: np.ndarray, image_points: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
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

        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = matcher.knnMatch(des_prev, des_new, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 4:
            print(f"Insufficient feature matches for triangulation: {len(good_matches)}")
            return [], []
        
        pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in good_matches])
        pts_new = np.float32([kp_new[m.trainIdx].pt for m in good_matches])
        
        P_prev = self.K @ np.hstack((R_prev, t_prev))
        P_new = self.K @ np.hstack((R_new, t_new))
        
        pts_prev_norm = cv2.undistortPoints(pts_prev.reshape(-1, 1, 2), self.K, None)
        pts_new_norm = cv2.undistortPoints(pts_new.reshape(-1, 1, 2), self.K, None)
        
        points_4d = cv2.triangulatePoints(P_prev, P_new, pts_prev_norm, pts_new_norm)
        points_3d = (points_4d[:3] / points_4d[3]).T
        
        new_points = []
        new_colors = []
        
        for i, pt in enumerate(points_3d):
            if pt[2] <= 0:
                continue
            
            pt_cam2 = R_new @ pt.reshape(3, 1) + t_new
            if pt_cam2[2, 0] <= 0:
                continue
            
            new_points.append(pt)
            new_colors.append([0.5, 0.5, 0.5])
        
        return new_points, new_colors
    
    def add_image(self, kp: List, des: np.ndarray, img_rgb: Optional[np.ndarray] = None) -> bool:
        if len(self.map_points) == 0:
            print("ERROR: Map is empty. Initialize with two-view reconstruction first.")
            return False
        
        prev_des = self.image_descriptors[-1]
        
        if prev_des is None or des is None:
            print("ERROR: Descriptors are None")
            return False
        
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = matcher.knnMatch(prev_des, des, k=2)        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        print(f"Matches with previous image: {len(good_matches)}")
        
        if len(good_matches) < 4:
            print("ERROR: Not enough matches for PnP")
            return False
        
        prev_kp = self.image_keypoints[-1]
        object_points = []
        image_points = []
        
        for m in good_matches:
            prev_idx = m.queryIdx
            curr_idx = m.trainIdx
            
            if prev_idx < len(self.map_points):
                object_points.append(self.map_points[prev_idx])
                image_points.append(kp[curr_idx].pt)
        
        if len(object_points) < 4:
            print("ERROR: Not enough 2D-3D correspondences")
            return False
        
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)
        
        print(f"Solving PnP with {len(object_points)} correspondences...")
        R_new, t_new, inliers = self.solve_pnp(object_points, image_points)
        
        if R_new is None:
            print("PnP pose estimation failed")
            return False
        
        print(f"✓ PnP pose estimated with {len(inliers) if inliers is not None else 0} inliers")
        
        self.camera_poses.append((R_new, t_new))
        self.image_keypoints.append(kp)
        self.image_descriptors.append(des)        
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
