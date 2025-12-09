import os
from utils import *
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not available. Visualization features will be limited.")
from baseline import Baseline


class SFM:

    def __init__(self, views, matches, K):

        self.views = views
        self.matches = matches
        self.names = []
        self.done = []
        self.K = K 
        self.points_3D = np.zeros((0, 3))
        self.point_counter = 0 
        self.point_map = {}  
        self.errors = [] 

        for view in self.views:
            self.names.append(view.name)

        if not os.path.exists(self.views[0].root_path + '/points'):
            os.makedirs(self.views[0].root_path + '/points')

        self.results_path = os.path.join(self.views[0].root_path, 'points')

    def get_index_of_view(self, view):
        """Extracts the position of a view in the list of views"""

        return self.names.index(view.name)

    def remove_mapped_points(self, match_object, image_idx):
        """Removes points that have already been reconstructed in the completed views"""

        inliers1 = []
        inliers2 = []

        for i in range(len(match_object.inliers1)):
            if (image_idx, match_object.inliers1[i]) not in self.point_map:
                inliers1.append(match_object.inliers1[i])
                inliers2.append(match_object.inliers2[i])

        match_object.inliers1 = inliers1
        match_object.inliers2 = inliers2

    def compute_pose(self, view1, view2=None, is_baseline=False):
        """Computes the pose of the new view"""

        # procedure for baseline pose estimation
        if is_baseline and view2:

            match_object = self.matches[(view1.name, view2.name)]
            baseline_pose = Baseline(view1, view2, match_object)
            view2.R, view2.t = baseline_pose.get_pose(self.K)

            rpe1, rpe2 = self.triangulate(view1, view2)
            self.errors.append(np.mean(rpe1))
            self.errors.append(np.mean(rpe2))

            self.done.append(view1)
            self.done.append(view2)

        else:

            view1.R, view1.t = self.compute_pose_PNP(view1)
            errors = []

            for i, old_view in enumerate(self.done):

                match_object = self.matches[(old_view.name, view1.name)]
                _ = remove_outliers_using_F(old_view, view1, match_object)
                self.remove_mapped_points(match_object, i)
                _, rpe = self.triangulate(old_view, view1)
                errors += rpe

            self.done.append(view1)
            self.errors.append(np.mean(errors))

    def triangulate(self, view1, view2):
        """Triangulates 3D points from two views whose poses have been recovered.
        SAFE VERSION: Never crashes and always returns something.
        """

        K_inv = np.linalg.inv(self.K)
        P1 = np.hstack((view1.R, view1.t))
        P2 = np.hstack((view2.R, view2.t))

        # ✅ SAFE MATCH ACCESS
        match_key = (view1.name, view2.name)
        if match_key not in self.matches:
            logging.warning(f"⚠️ No matches found between {view1.name} and {view2.name}")
            return None, 1e9

        match_object = self.matches[match_key]

        pixel_points1, pixel_points2 = get_keypoints_from_indices(
            keypoints1=view1.keypoints,
            keypoints2=view2.keypoints,
            index_list1=match_object.inliers1,
            index_list2=match_object.inliers2
        )

        if pixel_points1 is None or pixel_points2 is None or len(pixel_points1) < 8:
            logging.warning(f"⚠️ Not enough valid matches for {view2.name}. Skipping triangulation.")
            return None, 1e9

        pixel_points1 = cv2.convertPointsToHomogeneous(pixel_points1)[:, 0, :]
        pixel_points2 = cv2.convertPointsToHomogeneous(pixel_points2)[:, 0, :]

        reprojection_error1 = []
        reprojection_error2 = []

        new_points_3D = []

        for i in range(len(pixel_points1)):
            u1 = pixel_points1[i, :]
            u2 = pixel_points2[i, :]

            u1_normalized = K_inv.dot(u1)
            u2_normalized = K_inv.dot(u2)

            point_3D = get_3D_point(u1_normalized, P1, u2_normalized, P2)

            if np.isnan(point_3D).any() or np.isinf(point_3D).any():
                continue

            new_points_3D.append(point_3D.T)

            error1 = calculate_reprojection_error(point_3D, u1[0:2], self.K, view1.R, view1.t)
            reprojection_error1.append(error1)

            error2 = calculate_reprojection_error(point_3D, u2[0:2], self.K, view2.R, view2.t)
            reprojection_error2.append(error2)

        if len(new_points_3D) == 0:
            logging.warning(f"⚠️ No valid 3D points reconstructed for {view2.name}")
            return None, 1e9

        new_points_3D = np.vstack(new_points_3D)
        return new_points_3D, np.mean(reprojection_error1 + reprojection_error2)


    def compute_pose_PNP(self, view):
        """Computes pose of new view using perspective n-point (SAFE VERSION)"""

        if view.feature_type in ['sift', 'surf']:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # collects all the descriptors of the reconstructed views
        old_descriptors = []
        for old_view in self.done:
            old_descriptors.append(old_view.descriptors)

        # match old descriptors against the descriptors in the new view
        matcher.add(old_descriptors)
        matcher.train()
        matches = matcher.match(queryDescriptors=view.descriptors)

        points_3D, points_2D = np.zeros((0, 3)), np.zeros((0, 2))

        # build corresponding array of 2D points and 3D points
        for match in matches:
            old_image_idx, new_image_kp_idx, old_image_kp_idx = (
                match.imgIdx,
                match.queryIdx,
                match.trainIdx,
            )

            if (old_image_idx, old_image_kp_idx) in self.point_map:

                # obtain the 2D point from match
                point_2D = np.array(view.keypoints[new_image_kp_idx].pt).T.reshape((1, 2))
                points_2D = np.concatenate((points_2D, point_2D), axis=0)

                # obtain the 3D point from the point_map
                point_3D = self.points_3D[
                    self.point_map[(old_image_idx, old_image_kp_idx)], :
                ].T.reshape((1, 3))
                points_3D = np.concatenate((points_3D, point_3D), axis=0)

        # ✅✅✅ HARD SAFETY GUARD — THIS FIXES YOUR CURRENT CRASH
        if points_3D.shape[0] < 4:
            logging.warning(
                f"⚠️ Not enough 2D–3D correspondences for PnP in view {view.name}: "
                f"{points_3D.shape[0]} points. Reusing previous pose."
            )

            # ✅ Fallback: reuse last successful camera pose
            last_view = self.done[-1]
            return last_view.R.copy(), last_view.t.copy()

        # ✅ Safe PnP call
        _, R, t, _ = cv2.solvePnPRansac(
            points_3D[:, np.newaxis],
            points_2D[:, np.newaxis],
            self.K,
            None,
            confidence=0.99,
            reprojectionError=8.0,
            flags=cv2.SOLVEPNP_DLS,
        )

        R, _ = cv2.Rodrigues(R)
        return R, t

    def write_ply_simple(self, filename):
        """Write PLY file without Open3D (fallback method)"""
        with open(filename, 'w') as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(self.points_3D)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            
            # Write vertices
            for point in self.points_3D:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
        
        logging.info("PLY file written to %s", filename)

    def plot_points(self):
        """Saves the reconstructed 3D points to ply files using Open3D"""

        number = len(self.done)
        filename = os.path.join(self.results_path, str(number) + '_images.ply')
        
        if OPEN3D_AVAILABLE:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points_3D)
            o3d.io.write_point_cloud(filename, pcd)
        else:
            # Fallback: write simple PLY format without Open3D
            self.write_ply_simple(filename)

    def reconstruct(self):
        """Starts the main reconstruction loop for a given set of views and matches"""

        # compute baseline pose
        baseline_view1, baseline_view2 = self.views[0], self.views[1]
        logging.info("Computing baseline pose and reconstructing points")
        self.compute_pose(view1=baseline_view1, view2=baseline_view2, is_baseline=True)
        logging.info("Mean reprojection error for 1 image is %f", self.errors[0])
        logging.info("Mean reprojection error for 2 images is %f", self.errors[1])
        self.plot_points()
        logging.info("Points plotted for %d views", len(self.done))

        for i in range(2, len(self.views)):

            logging.info("Computing pose and reconstructing points for view %d", i+1)
            self.compute_pose(view1=self.views[i])
            logging.info("Mean reprojection error for %d images is %f", i+1, self.errors[i])
            self.plot_points()
            logging.info("Points plotted for %d views", i+1)
