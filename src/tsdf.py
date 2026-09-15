# Modified by Haochen Zhang, 2025
# Modified by 2024 Allen Ren, Princeton University
# Copyright (c) 2018 Andy Zeng
# Source: https://github.com/andyzeng/tsdf-fusion-python/blob/master/fusion.py
# BSD 2-Clause License

# Copyright (c) 2025, Honda Research Institute
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import cv2
import numpy as np
from numba import njit, prange
import random
import logging
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from skimage import measure
from sklearn.cluster import DBSCAN
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter, binary_erosion
from .geom import (
    points_in_circle,
    find_normal,
    close_operation,
    rigid_transform,
    run_dijkstra,
    fps,
)
import habitat_sim
from magnum import Vector3
from habitat_sim.utils.common import quat_to_magnum

from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis
from src.habitat import (
    pos_normal_to_habitat,
    pos_habitat_to_normal,
)

class TSDFPlanner:
    """Volumetric TSDF Fusion of RGB-D Images. No GPU mode.

    Add frontier-based exploration and semantic map.
    """

    def __init__(
        self,
        vol_bnds,
        voxel_size,
        floor_height_offset=0,
        pts_init=None,
        init_clearance=0,
        simulator=None
    ):
        """Constructor.
        Args:
          vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
            xyz bounds (min/max) in meters.
          voxel_size (float): The volume discretization in meters.
        """
        vol_bnds = np.asarray(vol_bnds)
        assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."
        assert (vol_bnds[:, 0] < vol_bnds[:, 1]).all()

        # Define voxel volume parameters
        self._vol_bnds = vol_bnds
        self._voxel_size = float(voxel_size)
        self._trunc_margin = 5 * self._voxel_size  # truncation on SDF
        self._color_const = 256 * 256
        self.simulator = simulator

        # Adjust volume bounds and ensure C-order contiguous
        self._vol_dim = (
            np.ceil((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size)
            .copy(order="C")
            .astype(int)
        )
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + self._vol_dim * self._voxel_size
        self._vol_origin = self._vol_bnds[:, 0].copy(order="C").astype(np.float32)

        # Initialize pointers to voxel volume in CPU memory
        # Assume all unobserved regions are occupied
        self._tsdf_vol_cpu = -np.ones(self._vol_dim).astype(np.float32)
        # for computing the cumulative moving average of observations per voxel
        self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

        # Semantic value
        self._val_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        self._weight_val_vol_cpu = np.zeros(self._vol_dim[:2]).astype(np.float32)

        # Room classification?
        self._region_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

        # Explored or not
        self._explore_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

        # Get voxel grid coordinates
        xv, yv, zv = np.meshgrid(
            range(self._vol_dim[0]),
            range(self._vol_dim[1]),
            range(self._vol_dim[2]),
            indexing="ij",
        )
        self.vox_coords = (
            np.concatenate(
                [xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)], axis=0
            )
            .astype(int)
            .T
        )

        # pre-compute
        self.cam_pts_pre = TSDFPlanner.vox2world(
            self._vol_origin, self.vox_coords, self._voxel_size
        )

        # Find the minimum height voxel
        self.min_height_voxel = int(floor_height_offset / self._voxel_size)

        # For masking the area around initial pose to be unoccupied
        coords_init = self.world2vox(pts_init)
        self.init_points = points_in_circle(
            coords_init[0],
            coords_init[1],
            int(init_clearance / self._voxel_size),
            self._vol_dim[:2],
        )

        self.target_point = None
        self.doorway_queue = []
        self.visited_doorways = set()
        self.current_target = None
        self.visited_voxels = []

    @staticmethod
    @njit(parallel=True)
    def vox2world(vol_origin, vox_coords, vox_size):
        """Convert voxel grid coordinates to world coordinates."""
        vol_origin = vol_origin.astype(np.float32)
        vox_coords = vox_coords.astype(np.float32)
        cam_pts = np.empty_like(vox_coords, dtype=np.float32)
        for i in prange(vox_coords.shape[0]):
            for j in range(3):
                cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j])
        return cam_pts

    @staticmethod
    @njit(parallel=True)
    def cam2pix(cam_pts, intr):
        """Convert camera coordinates to pixel coordinates."""
        intr = intr.astype(np.float32)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
        for i in prange(cam_pts.shape[0]):
            pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
            pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
        return pix

    def pix2cam(self, pix, intr):
        """Convert pixel coordinates to camera coordinates."""
        intr = intr.astype(np.float32)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        cam_pts = np.empty((pix.shape[0], 3), dtype=np.float32)
        for i in range(cam_pts.shape[0]):
            cam_pts[i, 2] = 1
            cam_pts[i, 0] = (pix[i, 0] - cx) / fx * cam_pts[i, 2]
            cam_pts[i, 1] = (pix[i, 1] - cy) / fy * cam_pts[i, 2]
        return cam_pts

    def world2vox(self, pts):
        pts = pts - self._vol_origin
        coords = np.round(pts / self._voxel_size).astype(int)
        coords = np.clip(coords, 0, self._vol_dim - 1)
        return coords

    @staticmethod
    @njit(parallel=True)
    def integrate_tsdf(tsdf_vol, dist, w_old, obs_weight):
        """Integrate the TSDF volume."""
        tsdf_vol_int = np.empty_like(tsdf_vol, dtype=np.float32)
        w_new = np.empty_like(w_old, dtype=np.float32)
        for i in prange(len(tsdf_vol)):
            w_new[i] = w_old[i] + obs_weight
            tsdf_vol_int[i] = (w_old[i] * tsdf_vol[i] + obs_weight * dist[i]) / w_new[i]
        return tsdf_vol_int, w_new

    def integrate_sem(
        self,
        sem_pix,
        radius=1.0,  # meter
        obs_weight=1.0,
    ):
        """Add semantic value to the 2D map by marking a circle of specified radius"""
        assert len(self.candidates) == len(sem_pix)
        for p_ind, p in enumerate(self.candidates):
            radius_vox = int(radius / self._voxel_size)
            pts = points_in_circle(p[0], p[1], radius_vox, self._vol_dim[:2])
            for pt in pts:
                w_old = self._weight_val_vol_cpu[pt[0], pt[1]].copy()
                self._weight_val_vol_cpu[pt[0], pt[1]] += obs_weight
                self._val_vol_cpu[pt[0], pt[1]] = (
                    w_old * self._val_vol_cpu[pt[0], pt[1]]
                    + obs_weight * sem_pix[p_ind]
                ) / self._weight_val_vol_cpu[pt[0], pt[1]]

    def integrate(
        self,
        color_im,
        depth_im,
        cam_intr,
        cam_pose,
        sem_im=None,
        w_new=None,
        obs_weight=1.0,
        margin_h=240,  # from top
        margin_w=120,  # each side,
        max_exp_depth=3.5 # added to count observation as explored
    ):
        """Integrate an RGB-D frame into the TSDF volume.
        Args:
          color_im (ndarray): An RGB image of shape (H, W, 3).
          depth_im (ndarray): A depth image of shape (H, W).
          cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
          cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
          sem_im (ndarray): An semantic image of shape (H, W).
          obs_weight (float): The weight to assign for the current observation. A higher
            value
          margin_h (int): The margin from the top of the image to exclude when integrating explored
          margin_w (int): The margin from the sides of the image to exclude when integrating explored
        """
        im_h, im_w = depth_im.shape

        # Fold RGB color image into a single channel image
        color_im = color_im.astype(np.float32)
        color_im = np.floor(
            color_im[..., 2] * self._color_const
            + color_im[..., 1] * 256
            + color_im[..., 0]
        )

        # Convert voxel grid coordinates to pixel coordinates
        cam_pts = rigid_transform(self.cam_pts_pre, np.linalg.inv(cam_pose))
        pix_z = cam_pts[:, 2]
        pix = TSDFPlanner.cam2pix(cam_pts, cam_intr)
        pix_x, pix_y = pix[:, 0], pix[:, 1]

        # Eliminate pixels outside view frustum
        valid_pix = np.logical_and(
            pix_x >= 0,
            np.logical_and(
                pix_x < im_w,
                np.logical_and(pix_y >= 0, np.logical_and(pix_y < im_h, pix_z > 0)),
            ),
        )
        depth_val = np.zeros(pix_x.shape)
        depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

        # narrow view
        valid_pix_narrow = np.logical_and(
            pix_x >= margin_w,
            np.logical_and(
                pix_x < (im_w - margin_w),
                np.logical_and(
                    pix_y >= margin_h,
                    np.logical_and(pix_y < im_h, pix_z > 0),
                ),
            ),
        )
        depth_val_narrow = np.zeros(pix_x.shape)
        depth_val_narrow[valid_pix_narrow] = depth_im[
            pix_y[valid_pix_narrow], pix_x[valid_pix_narrow]
        ]

        # Integrate TSDF
        depth_diff = depth_val - pix_z
        valid_pts = np.logical_and(depth_val > 0, depth_diff >= -self._trunc_margin)
        dist = np.maximum(-1, np.minimum(1, depth_diff / self._trunc_margin))
        valid_vox_x = self.vox_coords[valid_pts, 0]
        valid_vox_y = self.vox_coords[valid_pts, 1]
        valid_vox_z = self.vox_coords[valid_pts, 2]
        w_old = self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]

        depth_diff_narrow = depth_val_narrow - pix_z
        valid_pts_narrow = np.logical_and(
            depth_val_narrow > 0, depth_diff_narrow >= -self._trunc_margin
        )

        # NOTE: added
        within_range = depth_val_narrow < max_exp_depth
        valid_in_range = np.logical_and(valid_pts_narrow, within_range)

        '''valid_vox_x_narrow = self.vox_coords[valid_pts_narrow, 0]
        valid_vox_y_narrow = self.vox_coords[valid_pts_narrow, 1]
        valid_vox_z_narrow = self.vox_coords[valid_pts_narrow, 2]'''
        
        valid_vox_x_narrow = self.vox_coords[valid_in_range, 0]
        valid_vox_y_narrow = self.vox_coords[valid_in_range, 1]
        valid_vox_z_narrow = self.vox_coords[valid_in_range, 2]
        if w_new is None:
            tsdf_vals = self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            valid_dist = dist[valid_pts]
            tsdf_vol_new, w_new = TSDFPlanner.integrate_tsdf(
                tsdf_vals, valid_dist, w_old, obs_weight
            )
            self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
            self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new

            # Mark explored
            self._explore_vol_cpu[
                valid_vox_x_narrow, valid_vox_y_narrow, valid_vox_z_narrow
            ] = 1

            # Integrate color
            old_color = self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            old_b = np.floor(old_color / self._color_const)
            old_g = np.floor((old_color - old_b * self._color_const) / 256)
            old_r = old_color - old_b * self._color_const - old_g * 256
            new_color = color_im[pix_y[valid_pts], pix_x[valid_pts]]
            new_b = np.floor(new_color / self._color_const)
            new_g = np.floor((new_color - new_b * self._color_const) / 256)
            new_r = new_color - new_b * self._color_const - new_g * 256
            new_b = np.minimum(
                255.0, np.round((w_old * old_b + obs_weight * new_b) / w_new)
            )
            new_g = np.minimum(
                255.0, np.round((w_old * old_g + obs_weight * new_g) / w_new)
            )
            new_r = np.minimum(
                255.0, np.round((w_old * old_r + obs_weight * new_r) / w_new)
            )
            self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = (
                new_b * self._color_const + new_g * 256 + new_r
            )

        # Integrate semantics if specified
        if sem_im is not None:
            old_sem = self._val_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            new_sem = sem_im[pix_y[valid_pts], pix_x[valid_pts]]
            new_sem = (w_old * old_sem + obs_weight * new_sem) / w_new
            self._val_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = new_sem
        return w_new

    def get_volume(self):
        return self._tsdf_vol_cpu, self._color_vol_cpu

    def get_point_cloud(self):
        """Extract a point cloud from the voxel volume."""
        tsdf_vol, color_vol = self.get_volume()

        # Marching cubes
        # verts = measure.marching_cubes(tsdf_vol, level=0, method='lewiner')[0]
        # See: https://github.com/andyzeng/tsdf-fusion-python/issues/24
        verts = measure.marching_cubes(
            tsdf_vol, mask=np.logical_and(tsdf_vol > -0.5, tsdf_vol < 0.5), level=0
        )[0]
        verts_ind = np.round(verts).astype(int)
        verts = verts * self._voxel_size + self._vol_origin

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self._color_const)
        colors_g = np.floor((rgb_vals - colors_b * self._color_const) / 256)
        colors_r = rgb_vals - colors_b * self._color_const - colors_g * 256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)

        pc = np.hstack([verts, colors])
        return pc

    def get_mesh(self):
        """Compute a mesh from the voxel volume using marching cubes."""
        tsdf_vol, color_vol = self.get_volume()

        # Marching cubes
        # verts, faces, norms, vals = measure.marching_cubes(tsdf_vol, level=0, method='lewiner')
        # See: https://github.com/andyzeng/tsdf-fusion-python/issues/24
        verts, faces, norms, vals = measure.marching_cubes(
            tsdf_vol, mask=np.logical_and(tsdf_vol > -0.5, tsdf_vol < 0.5), level=0
        )
        verts_ind = np.round(verts).astype(int)
        verts = verts * self._voxel_size + self._vol_origin

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self._color_const)
        colors_g = np.floor((rgb_vals - colors_b * self._color_const) / 256)
        colors_r = rgb_vals - colors_b * self._color_const - colors_g * 256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)
        return verts, faces, norms, colors

    ############# For building semantic map and exploration #############
    def extract_walls(self):
        verts, faces, norms, colors = self.get_mesh()
        # Filter out ceiling/floor
        z = verts[:, 2]
        wall_pts = verts[(z > 0.1) & (z < 1.8)]  # filter based on height (adjustable)

        xy_pts = wall_pts[:, :2]  # (X, Y)
        clustering = DBSCAN(eps=1.0, min_samples=50).fit(xy_pts)
        labels = clustering.labels_
        unique_rooms = np.unique(labels[labels >= 0])

        room_boxes = []
        for room_id in unique_rooms:
            pts = xy_pts[labels == room_id].astype(np.float32)
            if len(pts) < 3:
                continue
            box = cv2.minAreaRect(pts)  # Or use cv2.boundingRect
            box_pts = cv2.boxPoints(box)  # (4, 2) corners
            room_boxes.append((room_id, box_pts))

    def cluster_doorways(self, doorways, unexp_w, eps=5, min_samples=3):
        doorway_xy = np.array([[v[0], v[1]] for v in doorways])  # (x, y)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(doorway_xy)
        labels = clustering.labels_

        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:
                continue
            clusters.setdefault(label, []).append(doorways[i])

        # Score clusters by length or count
        scored = []
        if len(clusters) > 0:
            max_size = max(len(v) for v in clusters.values())

        for label, cluster in clusters.items():
            cluster_arr = np.array(cluster)

            # score based on cluster size and spread
            span = np.max(cluster_arr[:, [0, 2]], axis=0) - np.min(cluster_arr[:, [0, 2]], axis=0)
            size_score = len(cluster)/max_size + 0.5 * np.linalg.norm(span)

            # calculate cluster center
            center = np.mean(cluster_arr, axis=0)
            dists = np.linalg.norm(cluster_arr - center, axis=1)
            center_voxel = cluster[np.argmin(dists)]  # closest voxel to centroid

             # Add weight to prioritize less explored regions
             # Get unexplored score in neighborhood (e.g. 5x5 region)
            k = 3  # half window size (change as needed)
            H, W = self._explore_vol_cpu.shape[:2]
            x, y = int(center_voxel[0]), int(center_voxel[2])  
            x1, x2 = max(x-k, 0), min(x+k+1, H)
            y1, y2 = max(y-k, 0), min(y+k+1, W)
            unexplored_patch = self._explore_vol_cpu[x1:x2, y1:y2]

            # Compute unexplored density: lower mean = more unexplored
            unexplored_score = 1.0 - np.mean(unexplored_patch) 

            # score based on distance from previously explored
            if len(self.visited_doorways) > 0 and len(self.visited_voxels) > 0:
                visited_pts = np.concatenate((np.array(list(self.visited_doorways))[:, :2], np.vstack([x.squeeze() for x in self.visited_voxels])))
                
                # Only consider (x, z) since that's how doorways are defined
                #explored_2d = visited_pts[:, [0, 2]]
                center_2d = np.array([center_voxel[0], center_voxel[2]])

                # Compute distances to all explored points
                dists_to_explored = np.linalg.norm(visited_pts - center_2d, axis=1)
                min_dist = np.min(dists_to_explored)

                # Normalize distance if needed (optional)
                dist_score = np.log(min_dist+1)  # or log(min_dist + 1), etc.
            else:
                dist_score = 0  # No explored points yet → no penalty

            score = size_score + unexp_w * unexplored_score + dist_score

            scored.append((score, center_voxel, cluster))

        # Sort clusters by score descending
        scored.sort(reverse=True, key=lambda x: x[0])
        #print("scored doorways: ", scored)
        center_voxels = np.array([center for _, center, _ in scored])
        scores = np.array([score for score, _, _ in scored])
        return scored, center_voxels, scores

    def detect_doorways_from_tsdf(self, pts, unexp_w, agent_height_m=1.5, tsdf_thresh=0.1):
        """
        Detect doorway candidates using a 2D horizontal slice of the TSDF volume.

        Args:
            tsdf_vol (np.ndarray): 3D TSDF grid (shape: [H, W, D]).
            origin (np.ndarray): Origin of the TSDF grid in world coordinates (3D).
            agent_height_m (float): Height at which to take the TSDF slice (e.g., agent's height).
            tsdf_thresh (float): Range to consider "free space" (e.g. -0.1 to 0.1).

        Returns:
            List of 3D world coordinates for candidate doorway points.
        """
        tsdf_vol = self._tsdf_vol_cpu #x, y, z
        W, D, H = tsdf_vol.shape # should be W, D, H

        # Compute voxel Y index for agent height slice
        height_idx = int((agent_height_m - self._vol_origin[1]) / self._voxel_size)
        height_idx = np.clip(height_idx, 1, H - 2)

        tsdf_slice = tsdf_vol[:, :, height_idx]  # shape: [W, D]

        doorways_voxel = []
        # REQUIRES THIS TO BE UPDATED
        # actually uses island to get connected region
        island, _ = self.get_island_around_pts(pts, height=0.4)
        unoccupied = island

        # erodes unoccupied map so camera isn't caught in object edges
        safe_unoccupied = binary_erosion(unoccupied, iterations=1)

        # narrow region of 3*2 voxels?
        door_thres = 5
        for i in range(1, W - door_thres):
            for j in range(1, D - door_thres):
                center_val = tsdf_slice[i, j]
                free = unoccupied[i, j]
                if not free:
                    continue  # not free space

                # in 2d grid (BEV)
                left = tsdf_slice[i, j - door_thres]
                right = tsdf_slice[i, j + door_thres]
                up = tsdf_slice[i - door_thres, j]
                down = tsdf_slice[i + door_thres, j]

                # TODO: check occupancy between i,j and left/right/up/down
                left_unocc = unoccupied[i, j - door_thres]
                right_unocc = unoccupied[i, j + door_thres]
                up_unocc = unoccupied[i - door_thres, j]
                down_unocc = unoccupied[i + door_thres, j]

                # Check for narrow corridor-like configuration
                is_x_narrow = not left_unocc and not right_unocc #(left > tsdf_thresh) and (right > tsdf_thresh)
                is_z_narrow = not up_unocc and not down_unocc #(up > tsdf_thresh) and (down > tsdf_thresh)

                if (is_x_narrow or is_z_narrow) and safe_unoccupied[i, j]:
                    doorways_voxel.append((i, j, height_idx))

        #logging.info(f"doorways_voxel: {doorways_voxel}")
        if len(doorways_voxel) > 0:
            # cluster detected doorways
            scored_clusters, center_voxels, scores = self.cluster_doorways(doorways_voxel, unexp_w)

            return doorways_voxel, center_voxels, scores, island
        else:
            return [], [], [], []

    def in_current_view(
        self,
        pts,
        im_w,
        im_h,
        cam_intr,
        cam_pose,
        height=1.5,
        point_min_dist=2,
        point_max_dist=10,
        cam_offset=0.5,
        **kwargs,
        ):

        cur_point = self.world2vox(pts)
        island, unoccupied = self.get_island_around_pts(pts, height=height)
        unexplored = (np.sum(self._explore_vol_cpu, axis=-1) == 0).astype(int)
        for point in self.init_points:
            unexplored[point[0], point[1]] = 0
        occupied = np.logical_not(unoccupied).astype(int)
        cam_pose = cam_pose @ np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, cam_offset],
                [0, 0, 0, 1],
            ]
        )
        mask = self.get_current_view_mask(
            cam_intr, cam_pose, im_w, im_h, slack=0, margin_h=100, margin_w=30
        )

        ############## Get unoccupied reachable points in view ##############

        # Mask the unoccupied region to be only the current view
        unoccupied_in_view = np.multiply(unoccupied, mask)
        unoccupied_reachable_in_view = np.argwhere((island) & (unoccupied_in_view))

        # Subsample - weigh closer points more
        '''if len(unoccupied_reachable_in_view) > 0:
            subsample_inds = np.random.choice(
                range(len(unoccupied_reachable_in_view)),
                min(num_max_unoccupied, len(unoccupied_reachable_in_view)),
                replace=False,
            )
            unoccupied_reachable_in_view = unoccupied_reachable_in_view[subsample_inds]
        else:
            unoccupied_reachable_in_view = np.empty((0, 2))'''

        # Check unoccupied between point and current point - skip if any occupied
        unoccupied_reachable_in_view_new = np.empty((0, 2))
        for point in unoccupied_reachable_in_view:
            if not self.check_occupied_between(point, cur_point, occupied, threshold=1):
                unoccupied_reachable_in_view_new = np.concatenate(
                    (unoccupied_reachable_in_view_new, [point]), axis=0
                )
        unoccupied_reachable_in_view = unoccupied_reachable_in_view_new

        # Only keep points within desired range
        if len(unoccupied_reachable_in_view) > 0:
            dist_all = np.sqrt(
                (unoccupied_reachable_in_view[:, 0] - cur_point[0]) ** 2
                + (unoccupied_reachable_in_view[:, 1] - cur_point[1]) ** 2
            )
            unoccupied_reachable_in_view = unoccupied_reachable_in_view[
                (dist_all > point_min_dist / self._voxel_size)
                & (dist_all < point_max_dist / self._voxel_size)
            ]
            dist_all = dist_all[
                (dist_all > point_min_dist / self._voxel_size)
                & (dist_all < point_max_dist / self._voxel_size)
            ]
        
        return unoccupied_reachable_in_view
    

    def find_next_pose(
        self,
        pts,
        angle,
        flag_no_val_weight=False,
        unexplored_T=0.5,
        unoccupied_T=3,
        val_T=0.5,
        val_dir_T=0.5,
        dist_T=10,
        min_dist_from_cur=0.5,
        max_dist_from_cur=3,
        frontier_spacing=1.5,
        frontier_min_neighbors=3,
        frontier_max_neighbors=4,
        max_unexplored_check_frontier=3.0,
        max_unoccupied_check_frontier=1.0,
        max_val_check_frontier=5.0,
        smooth_sigma=5,
        eps=0.5,
        **kwargs,
    ):
        """Determine the next frontier to traverse to with semantic-value-weighted sampling."""
        cur_point = self.world2vox(pts)
        if hasattr(self, "cur_point"):
            island = self.island
            unoccupied, occupied = self.unoccupied, self.occupied
            unexplored, unexplored_neighbors = (
                self.unexplored,
                self.unexplored_neighbors,
            )
        else:
            island, unoccupied = self.get_island_around_pts(pts, height=0.4)
            occupied = np.logical_not(unoccupied).astype(int)
            unexplored = (np.sum(self._explore_vol_cpu, axis=-1) == 0).astype(int)
            for point in self.init_points:
                unexplored[point[0], point[1]] = 0
            kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
            unexplored_neighbors = ndimage.convolve(
                unexplored, kernel, mode="constant", cval=0.0
            )
        self.unexplored_neighbors = unexplored_neighbors
        self.unoccupied = unoccupied

        # get semantic map by taking max over z
        val_vol_2d = np.max(self._val_vol_cpu, axis=2).copy()

        # smoothen the map
        val_vol_2d = gaussian_filter(val_vol_2d, sigma=smooth_sigma)

        # add erosion
        safe_island = binary_erosion(island, iterations=1)
        
        # use nonzero_val_vol quantile for frontier normalization
        frontiers = np.argwhere(
            safe_island
            & (unexplored_neighbors >= frontier_min_neighbors)
            & (unexplored_neighbors <= frontier_max_neighbors)
        )

        # check curr vox in frontiers - should not happen
        if len(frontiers) > 0:
            mask = ~(frontiers == cur_point[:2]).all(axis=1)
            frontiers = frontiers[mask]

        frontiers_pre_cluster = frontiers.copy()

        # Fit frontiers
        if len(frontiers) > 10:
            db = DBSCAN(eps=eps, min_samples=2).fit(frontiers)
            labels = db.labels_
            # get one point from each cluster
            frontiers_new = np.empty((0, 2))
            for label in np.unique(labels):
                if label == -1:
                    continue
                cluster = frontiers[labels == label]
                # take the one that is closest to mean
                dist = np.sqrt(
                    (cluster[:, 0] - np.mean(cluster[:, 0])) ** 2
                    + (cluster[:, 1] - np.mean(cluster[:, 1])) ** 2
                )
                center = cluster[np.argmin(dist)]
                frontiers_new = np.append(frontiers_new, [center], axis=0)
            frontiers = frontiers_new.astype(int)

        # subsample
        frontiers_weight = np.zeros((len(frontiers)))

        # Commit
        point_type = "current"
        if self.target_point is None:

            # Get weights for frontiers
            frontiers_weight = np.empty((0))
            frontiers_new = np.empty((0, 2))
            # start_time = time.time(
            for point in frontiers:

                # find normal into unexplored
                normal = self.find_normal_into_space(point, unexplored, unexplored)

                # Then check how much unoccupied in that direction
                max_pixel_check = int(max_unoccupied_check_frontier / self._voxel_size)
                dir_pts = np.round(
                    point + np.arange(max_pixel_check)[:, np.newaxis] * normal
                ).astype(int)
                dir_pts = self.clip_2d_array(dir_pts)
                unoccupied_rate = (
                    np.sum(unoccupied[dir_pts[:, 0], dir_pts[:, 1]] == 1)
                    / max_pixel_check
                )

                # Check the radio of unexplored in the direction, until hits obstacle
                max_pixel_check = int(max_unexplored_check_frontier / self._voxel_size)
                dir_pts = np.round(
                    point + np.arange(max_pixel_check)[:, np.newaxis] * normal
                ).astype(int)
                dir_pts = self.clip_2d_array(dir_pts)
                unexplored_rate = (
                    np.sum(unexplored[dir_pts[:, 0], dir_pts[:, 1]] == 1)
                    / max_pixel_check
                )

                # Check value in the direction
                max_pixel_check = int(max_val_check_frontier / self._voxel_size)
                dir_pts = np.round(
                    point + np.arange(max_pixel_check)[:, np.newaxis] * normal
                ).astype(int)
                dir_pts = self.clip_2d_array(dir_pts)
                val_vol_2d_dir = val_vol_2d[dir_pts[:, 0], dir_pts[:, 1]]
                # keep non zero value only
                val_vol_2d_dir = val_vol_2d_dir[val_vol_2d_dir > 0]
                if len(val_vol_2d_dir) == 0:
                    val = 0
                else:
                    val = np.mean(val_vol_2d_dir)

                # Get weight - unexplored, unoccupied, and value
                weight = np.exp(unexplored_rate / unexplored_T)  # [0-1] before T
                weight *= np.exp(unoccupied_rate / unoccupied_T)  # [0-1] before T
                if not flag_no_val_weight:
                    weight *= np.exp(
                        val_vol_2d[point[0], point[1]] / val_T
                    )  # [0-1] before T
                    weight *= np.exp(val / val_dir_T)  # [0-1] before T

                # Check distance to current point - make weight very small if too close and aligned
                dist = (
                    np.sqrt(
                        (cur_point[0] - point[0]) ** 2 + (cur_point[1] - point[1]) ** 2
                    )
                    * self._voxel_size
                )
                pts_angle = np.arctan2(normal[1], normal[0]) - np.pi / 2
                weight *= np.exp(-dist / dist_T)
                if (
                    dist < min_dist_from_cur / self._voxel_size
                    and np.abs(angle - pts_angle) < np.pi / 6
                ):
                    weight *= 1e-3

                # Save weight
                frontiers_weight = np.append(frontiers_weight, weight)
                frontiers_new = np.concatenate((frontiers_new, [point]), axis=0)
            frontiers = frontiers_new.astype(int)
            logging.info(f"Number of frontiers for next pose: {len(frontiers)}")

            # raise frontier value if there is frontier
            if len(frontiers) > 0:
                logging.info(
                    f"Mean and std of frontier weight: {np.mean(frontiers_weight):.3f},"
                    f" {np.std(frontiers_weight):.3f}"
                )
                point_type = "frontier"

                # take best point until it satisfies condition
                max_try = 50
                cnt_try = 0
                while 1:
                    cnt_try += 1
                    if cnt_try > max_try:
                        point_type = "current"
                        break
                    frontiers_weight_red = frontiers_weight / np.mean(
                        frontiers_weight
                    )  # prevent overflowing
                    frontier_ind = np.random.choice(
                        range(len(frontiers)),
                        p=frontiers_weight_red / np.sum(frontiers_weight_red),
                    )
                    logging.info(f"weight: {frontiers_weight[frontier_ind]:.3f}")
                    max_point = frontiers[frontier_ind]

                    # find the direction into unexplored
                    direction = self.find_normal_into_space(
                        max_point, unexplored, unexplored
                    )

                    # Move back in the opposite direction of the normal by spacing, so the robot can see the frontier
                    # there is a chance that the point is outside the free space
                    next_point = np.array(max_point, dtype=float)
                    max_backtrack = int(frontier_spacing / self._voxel_size)
                    min_backtrack = 2
                    num_backtrack = 0
                    while 1:
                        next_point -= direction
                        num_backtrack += 1
                        if num_backtrack >= max_backtrack:
                            break

                        # break if close to boundary
                        if not self.check_within_bnds(next_point):
                            break

                        # break if occupied
                        if (
                            occupied[int(next_point[0]), int(next_point[1])]
                            or not island[int(next_point[0]), int(next_point[1])]
                        ):
                            next_point += 2 * direction
                            break
                    next_point = np.round(next_point).astype(int)
                    if (
                        num_backtrack >= min_backtrack
                        and self.check_within_bnds(next_point)
                        and island[int(next_point[0]), int(next_point[1])]
                    ):
                        break  # stop searching

            # no patch used
            if point_type == "current":
                logging.info("No patches, return current point and random direction")
                max_point = cur_point[:2]
                next_point = cur_point[:2]
                direction = np.random.rand(2) - 0.5
                direction = direction / np.linalg.norm(direction)
        else:
            point_type = "commit"
            next_point = self.target_point.copy()
            direction = self.target_direction.copy()
            max_point = self.max_point.copy()
        logging.info(f"Next pose type: {point_type}")
        logging.info(f"Next pos: {next_point}")

        # Check if the point is beyond the max dist. Note that not using dist from dijkstra for saving time, then not taking account into obstacles when calculating distance
        dist = np.sqrt(
            (next_point[0] - cur_point[0]) ** 2 + (next_point[1] - cur_point[1]) ** 2
        )
        if dist > max_dist_from_cur / self._voxel_size:
            self.target_point = next_point.copy()
            self.target_direction = direction.copy()
            self.max_point = max_point.copy()

            island_free = np.logical_not(island)  # 0 for free
            path = run_dijkstra(island_free, cur_point, next_point)
            max_num = min(int(max_dist_from_cur / self._voxel_size), len(path) - 1)
            next_point = np.array(path[max_num])
            direction = max_point - next_point  # direction to the max point
            direction = direction / np.linalg.norm(direction)
            logging.info(
                f"Current {cur_point[:2]}, target {self.target_point}, move to"
                f" {next_point}"
            )
        if dist <= max_dist_from_cur / self._voxel_size or max_num == len(path) - 1:
            self.target_point = None
            self.target_direction = None
            self.max_point = None

        # Plot
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 18))
        ax1.imshow(unoccupied)
        ax1.scatter(max_point[1], max_point[0], c="r", s=30, label="max")
        ax1.scatter(cur_point[1], cur_point[0], c="b", s=30, label="current")
        ax1.scatter(next_point[1], next_point[0], c="g", s=30, label="actual")
        ax1.set_title("Unoccupied")
        ax2.imshow(island)
        ax2.set_title("Island")
        ax3.imshow(unexplored_neighbors)
        for point in frontiers_pre_cluster:
            ax3.scatter(point[1], point[0], color="white", s=20, alpha=1)
        ax3.set_title("Unexplored neighbors")
        im = ax4.imshow(val_vol_2d)
        for point in frontiers:
            ax4.scatter(point[1], point[0], color="white", s=20, alpha=1)
        fig.colorbar(im, orientation="vertical", ax=ax4, fraction=0.046, pad=0.04)
        ax4.scatter(max_point[1], max_point[0], c="r", s=30, label="max")
        ax4.scatter(cur_point[1], cur_point[0], c="b", s=30, label="current")
        ax4.scatter(next_point[1], next_point[0], c="g", s=30, label="actual")
        ax4.quiver(
            next_point[1],
            next_point[0],
            direction[1],
            direction[0],
            color="r",
            scale=5,
            angles="xy",
            alpha=0.2,
        )
        ax4.set_title("Current sem values")
        im = ax5.imshow(island)
        ax5.set_title("Path on island")
        frontier_weights = np.zeros_like(val_vol_2d)
        for point, weight in zip(frontiers, frontiers_weight):
            frontier_weights[point[0], point[1]] = weight
        im = ax6.imshow(frontier_weights)
        fig.colorbar(im, orientation="vertical", ax=ax6, fraction=0.046, pad=0.04)
        ax6.scatter(max_point[1], max_point[0], c="r", s=20, label="max")
        ax6.set_title("Frontier weights")

        # Convert back to world coordinates
        next_point_normal = next_point * self._voxel_size + self._vol_origin[:2]
        
        rel_dir = next_point - cur_point[:2]
        logging.info(f"rel dir: {rel_dir}")
        if rel_dir[0] == 0 and rel_dir[1] == 0:
            # turn 180
            next_rel_yaw = angle + np.pi
        else:
            rel_dir = rel_dir / np.linalg.norm(rel_dir)
            logging.info(f"rel dir: {rel_dir}")
            next_rel_yaw = np.arctan2(rel_dir[1], rel_dir[0])
        
        next_yaw = next_rel_yaw # NOTE: CHANGED

        return next_point_normal, next_yaw, next_point, fig, unoccupied

    def plot(self, pts, height=0.4):
        fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, figsize=(20, 18))
        
        island, unoccupied = self.get_island_around_pts(pts, height=0.4)
        ax1.imshow(unoccupied)
        ax1.set_title("Current and next steps")

        safe_unoccupied = binary_erosion(unoccupied, iterations=1)
        ax2.imshow(safe_unoccupied)
        ax2.set_title("Eroded unoccupied space")
        
        im = ax3.imshow(island)
        ax3.set_title("Path on island")
        
        # agent height?
        H, W = self._explore_vol_cpu.shape[:2]
        height_idx = int((height - self._vol_origin[1]) / self._voxel_size)
        height_idx = np.clip(height_idx, 1, H - 2)
        
        ax4.imshow(self._explore_vol_cpu[:, :, 0])
        ax4.set_title("Explored space")
        return fig, unoccupied

    def pix_to_voxel(self, pts_normal, pt):
        vox_coord = self.world2vox(pt)
        # vox coord is 3d?
        
        island, unoccupied = self.get_island_around_pts(pts_normal, height=0.4)

        # find nearest unoccupied that is reachable
        safe_unoccupied = binary_erosion(island, iterations=1)
        occupied_mask = np.logical_not(safe_unoccupied).astype(np.uint8)

        # Compute distance transform: for each voxel, gives distance to nearest unoccupied voxel
        # Also returns the index of the nearest unoccupied voxel for each voxel
        # TODO: use unoccupied instead?
        dist, indices = distance_transform_edt(occupied_mask, return_indices=True)
        #print(indices.shape)

        # Retrieve the index of the nearest unoccupied voxel from the precomputed index map
        nearest = np.array(indices[:, vox_coord[0], vox_coord[1]])
        nearest_3d = np.array([nearest[0], nearest[1], 0], dtype=np.float32)  # shape: (3,)
        nearest_3d = nearest_3d.reshape(1, 3) 
        print(nearest_3d)
        #print(occupied_mask.shape)
        logging.info(f"Island val of nearest voxel: {island[int(nearest_3d[0][0]), int(nearest_3d[0][1])]}")

        next_pos = TSDFPlanner.vox2world(self._vol_origin, nearest_3d, self._voxel_size)[0]
        
        self.island, self.unoccupied = island, unoccupied
        #print("next pos", next_pos)
        return next_pos, nearest_3d[0][:2]
    
    def is_view_obstructed(self, start_vox, angle, agent, camera_tilt, threshold=1):
        agent_height_m = 1.5
        W, D, H = self._tsdf_vol_cpu.shape

        height_idx = int(agent_height_m/self._voxel_size + self.min_height_voxel)
        
        vox_3d = np.append(start_vox, height_idx)
        start_vox = np.array(vox_3d)

        #start_pos = TSDFPlanner.vox2world(self._vol_origin, start_vox, self._voxel_size)[0]

        # Direction vector from yaw + pitch
        dir_x = np.cos(camera_tilt) * np.cos(angle)
        dir_y = np.sin(camera_tilt)
        dir_z = np.cos(camera_tilt) * np.sin(angle)
        direction = np.array([dir_x, dir_z, dir_y])

        # Step along ray in small increments (meters)
        step_size = self._voxel_size / 2  # half-voxel increment
        step_size = 0.5
        num_steps = int(threshold / step_size)

        print("Raycast from: ", start_vox)
        print("tsdf vol shape: ", self._tsdf_vol_cpu.shape)
        for i in range(1, num_steps + 1):
            # one voxel at a time
            point_vox = start_vox + direction * i

            # If point outside TSDF bounds, stop
            #if not self.is_voxel_in_bounds(point_vox):
            #    break

            print("point vox: ", point_vox[0], point_vox[1], point_vox[2])
            if point_vox[0] > W or point_vox[1] > D or point_vox[2] > H:
                print("Past bounds")
                return False, None
            # Check occupancy from TSDF (less than 0)
            if self._tsdf_vol_cpu[int(point_vox[0]), int(point_vox[1]), int(point_vox[2])] <= 0:
                dist = np.linalg.norm(point_vox - start_vox)
                # threshold in voxel space
                if dist < threshold:
                    print(f"Hit dist: {dist}")
                    return True, dist  # too close

        return False, None  # no obstacle within range
    
    def _is_doorway_explored(self, pt, threshold=2, agent_height_m=1.5):
        """
        Check if a doorway voxel has been seen at least 'threshold' times.
        """
        #voxel_idx = ((np.array(pt) - self._origin) / self._voxel_size).astype(int)
        x, y = pt[0], pt[1]
        W, D, H = self._tsdf_vol_cpu.shape # should be W, D, H?

        # Compute voxel Y index for agent height slice
        height_idx = int((agent_height_m - self._vol_origin[1]) / self._voxel_size)
        height_idx = np.clip(height_idx, 1, H - 2)

        shape = self._explore_vol_cpu.shape
        if 0 <= x < shape[0] and 0 <= y < shape[1]:
            return self._explore_vol_cpu[x, y, height_idx] >= threshold
        return False

    def _is_doorway_visited(self, pt, threshold=6):
        # TODO: check if point is visited in general, not just amongst visited doorways
        # relative to voxel space
        for v in self.visited_doorways:
            # if within 5 voxels, count as visited
            if np.linalg.norm(np.array(pt) - np.array(v)) < threshold:
                return True
        
        # check visited steps
        for v in self.visited_voxels:
            if np.linalg.norm(np.array(pt[:2]) - np.array(v)) < threshold/2:
                return True

        return False

    # ALTERNATE
    def find_next_pose_doorway(
        self,
        pts, # agent position in world coordinates
        unexp_w,
        angle,
        agent,
        camera_tilt,
        max_dist_from_cur=3,
        use_find_normal=False,
        **kwargs,
    ):
        """
        Choose the next exploration goal based on doorway-driven frontier selection.
        """
        agent_vox = self.world2vox(pts)
        island, unoccupied = self.get_island_around_pts(pts)
        if island is None:
            print("[TSDF] No reachable area.")
            return None, None, None, None
        self.island = island
        
        # update explored to get unexplored neighbors
        occupied = np.logical_not(unoccupied).astype(int)
        unexplored = (np.sum(self._explore_vol_cpu, axis=-1) == 0).astype(int)
        for point in self.init_points:
            unexplored[point[0], point[1]] = 0
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        unexplored_neighbors = ndimage.convolve(
            unexplored, kernel, mode="constant", cval=0.0
        )
        self.unexplored_neighbors = unexplored_neighbors
        self.unoccupied = unoccupied

        # If we don't have any current target, look for more doorway points
        if self.current_target is None:
            # Detect new doorway points (global or FOV-based)
            new_doorways, center_voxels, _, _ = self.detect_doorways_from_tsdf(pts, unexp_w)
            # NOTE: CHECK clear?
            self.doorway_queue = []

            # Add only unvisited doorways
            # NOTE: CHANGED TO CLUSTERS
            for dw in center_voxels: #new_doorways:
                if not self._is_doorway_visited(dw) and tuple(dw) not in self.doorway_queue:
                    self.doorway_queue.append(tuple(dw))

            # Pop the next unexplored target
            while self.doorway_queue:
                next_dw = self.doorway_queue.pop(0)
                if not self._is_doorway_visited(next_dw):
                    self.current_target = next_dw
                    break

            if self.current_target is None:
                print("[TSDF] No more doorway targets!")
                return None, None, None, None

        # Step toward the current target (in voxel space)
        curr_vox = self.world2vox(pts)
        direction = np.array(self.current_target) - np.array(curr_vox)
        dist = np.linalg.norm(direction[:2])
        direction = direction / (dist + 1e-8) # normalized direction

        # distance is in voxel space
        if dist < 5:
            print("[TSDF] Already at doorway target.")
            self.visited_doorways.add(tuple(self.current_target))
            # detect doorways again as island may have changed
            new_doorways, center_voxels, _, _ = self.detect_doorways_from_tsdf(pts, unexp_w)
            self.doorway_queue = []

            # Add only unvisited doorways
            for dw in center_voxels: #new_doorways:
                if not self._is_doorway_visited(dw) and tuple(dw) not in self.doorway_queue:
                    self.doorway_queue.append(tuple(dw))
            
            #return None, None, None
            while self.doorway_queue:
                next_dw = self.doorway_queue.pop(0)
                if not self._is_doorway_visited(next_dw):
                    self.current_target = next_dw
                    break

            if self.current_target is None:
                print("[TSDF] No more doorway targets!")
                return None, None, None
            
            direction = np.array(self.current_target) - np.array(curr_vox)
            dist = np.linalg.norm(direction[:2])
            direction = direction / (dist + 1e-8) # normalized direction

        # If within max_dist, step directly to target
        if dist <= max_dist_from_cur / self._voxel_size:
            logging.info(f"Move to doorway target: {self.current_target}")
            next_vox = self.current_target[:2]
            self.visited_doorways.add(tuple(self.current_target))
            self.current_target = None
        else:
            island_free = np.logical_not(island)  # turn it into 0 is free?

            # erode
            island_free_erode = binary_erosion(island_free, iterations=1)

            path = run_dijkstra(island_free, curr_vox, self.current_target)
            # dijkstra wants grid where 0 is free
            max_num = min(int(max_dist_from_cur / self._voxel_size), len(path) - 1)
            next_vox = np.array(path[max_num])
            logging.info(f"island val at next: {island_free[next_vox[0], next_vox[1]]}")
            
            logging.info(
                f"Current {curr_vox[:2]}, target {self.current_target}, move to"
                f" {next_vox}"
            )
            print("Stepping to doorway target: ", self.current_target)
            #next_vox = np.array(curr_vox) + direction * max_dist_from_cur / self._voxel_size

        island_val = island[next_vox[0], next_vox[1]]
        print("island val at next: ", island_val)
        next_vox = np.array([next_vox])
        next_pos = TSDFPlanner.vox2world(self._vol_origin, next_vox, self._voxel_size)[0]

        # TODO: test whether using find_normal_into_space instead works better
        if use_find_normal:
            unexplored = (np.sum(self._explore_vol_cpu, axis=-1) == 0).astype(int)
            normal = self.find_normal_into_space(next_vox[0], island, occupied)
            print("normal: ", normal)
            print("direction: ", direction)
            direction = normal

        next_yaw = np.arctan2(direction[1], direction[0])

        obstructed, dist = self.is_view_obstructed(next_vox, next_yaw, agent, camera_tilt)
        if obstructed:
            logging.info(f"Next view obstructed! dist: {dist}")

        self.visited_voxels.append(next_vox)
        
        print("next pos, ", next_pos)
        return next_pos, next_yaw, next_vox[0][:2], island


    def get_island_around_pts(self, pts, fill_dim=0.4, height=0.4):
        """Find the empty space around the point (x,y,z) in the world frame"""
        # Convert to voxel coordinates
        cur_point = self.world2vox(pts)

        # Check if the height voxel is occupied
        height_voxel = int(height / self._voxel_size) + self.min_height_voxel
        unoccupied = np.logical_and(
            self._tsdf_vol_cpu[:, :, height_voxel] > 0, self._tsdf_vol_cpu[:, :, 0] < 0
        )  # check there is ground below

        # Set initial pose to be free
        for point in self.init_points:
            unoccupied[point[0], point[1]] = 1

        # filter small islands smaller than size 2x2 and fill in gap of size 2
        fill_size = int(fill_dim / self._voxel_size)
        structuring_element_close = np.ones((fill_size, fill_size)).astype(bool)
        unoccupied = close_operation(unoccupied, structuring_element_close)

        # Find the connected component closest to the current location is, if the current location is not free
        # this is a heuristic to determine reachable space, although not perfect
        islands = measure.label(unoccupied, connectivity=1)
        if unoccupied[cur_point[0], cur_point[1]] == 1:
            islands_ind = islands[cur_point[0], cur_point[1]]  # use current one
        else:
            # find the closest one - tbh, this should not happen, but it happens when the robot cannot see the space immediately in front of it because of camera height and fov
            y, x = np.ogrid[: unoccupied.shape[0], : unoccupied.shape[1]]
            dist_all = np.sqrt((x - cur_point[1]) ** 2 + (y - cur_point[0]) ** 2)
            dist_all[islands == islands[cur_point[0], cur_point[1]]] = np.inf
            island_coords = np.unravel_index(np.argmin(dist_all), dist_all.shape)
            islands_ind = islands[island_coords[0], island_coords[1]]
        island = islands == islands_ind
        return island, unoccupied

    def get_current_view_mask(
        self,
        cam_intr,
        cam_pose,
        im_w,
        im_h,
        slack=0,
        margin_h=0,
        margin_w=0,
    ):
        cam_pts = rigid_transform(self.cam_pts_pre, np.linalg.inv(cam_pose))
        pix_z = cam_pts[:, 2]
        pix = TSDFPlanner.cam2pix(cam_pts, cam_intr)
        pix_x, pix_y = pix[:, 0], pix[:, 1]
        valid_pix = np.logical_and(
            pix_x >= -slack + margin_w,
            np.logical_and(
                pix_x < (im_w + slack - margin_w),
                np.logical_and(
                    pix_y >= -slack + margin_h,
                    np.logical_and(pix_y < im_h + slack, pix_z > 0),
                ),
            ),
        )
        # make a 2D mask where valid pix is 1 and 0 otherwise
        valid_pix = valid_pix.reshape(self._vol_dim).astype(int)
        mask = np.max(valid_pix, axis=2)  # take the max over height (z)
        return mask

    def check_occupied_between(self, p1, p2, occupied, threshold):
        direction = np.array([p2[0] - p1[0], p2[1] - p1[1]]).astype(float)
        num_points = int(np.linalg.norm(direction))
        dir_norm = direction / np.linalg.norm(direction)
        points_between = (
            p1[:2] + dir_norm * np.arange(num_points + 1)[:, np.newaxis]
        ).astype(int)
        points_occupied = np.sum(occupied[points_between[:, 0], points_between[:, 1]])
        return points_occupied > threshold

    def check_within_bnds(self, pts, slack=0):
        return not (
            pts[0] <= slack
            or pts[0] >= self._vol_dim[0] - slack
            or pts[1] <= slack
            or pts[1] >= self._vol_dim[1] - slack
        )

    def clip_2d_array(self, array):
        return array[
            (array[:, 0] >= 0)
            & (array[:, 0] < self._vol_dim[0])
            & (array[:, 1] >= 0)
            & (array[:, 1] < self._vol_dim[1])
        ]

    def find_normal_into_space(self, point, island, space, num_check=10):
        """Find the normal direction into the space"""
        normal = find_normal(
            island.astype(int), point[0], point[1]
        )  # but normal is ambiguous, so need to find which direction is unoccupied
        dir_1 = (point + np.arange(num_check)[:, np.newaxis] * normal).astype(int)
        dir_2 = (point - np.arange(num_check)[:, np.newaxis] * normal).astype(int)
        dir_1 = self.clip_2d_array(dir_1)
        dir_2 = self.clip_2d_array(dir_2)
        dir_1_occupied = np.sum(space[dir_1[:, 0], dir_1[:, 1]])
        dir_2_occupied = np.sum(space[dir_2[:, 0], dir_2[:, 1]])
        direction = normal
        if dir_1_occupied < dir_2_occupied:
            direction *= -1
        elif dir_1_occupied == dir_2_occupied:  # randomly choose one
            if random.random() < 0.5:
                direction *= -1
        return direction