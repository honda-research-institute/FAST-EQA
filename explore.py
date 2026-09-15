import os
import cv2
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.distance import pdist
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import pil_to_tensor
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from src.habitat import (
    make_simple_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
)
from src.geom import get_cam_intr, get_scene_bnds, pixel2world
from am_radio import AMRadio

class ExplorationAgent():
    def __init__(self, cfg, vlm, featurizer, tsdf_planner, agent, episode_data_dir, floor_height, gpu_id=0):
        self.cfg = cfg
        self.device = f"cuda:{gpu_id}"
        self.vlm = vlm
        self.tsdf_planner = tsdf_planner 
        self.featurizer = featurizer
        self.agent = agent

        self.episode_data_dir = episode_data_dir
        self.max_det_rooms = cfg.max_det_rooms
        self.partial_step_ratio = cfg.partial_step_ratio
        self.img_height = cfg.img_height
        self.img_width = self.cfg.img_width
        self.cam_intr = get_cam_intr(cfg.hfov, self.img_height, self.img_width)
        self.floor_height = floor_height
        self.unexp_w = cfg.planner.unexp_w
        self.use_find_normal = cfg.use_find_normal
        self.curr_turns = 0
        self.past_steps = []
        self.local_queue = []
        self.check_stop = False
        self.prev_global_state = None
        self.local = False

    # TODO: convert to 3d coords and select view
    def detect_obj(self, ind, image, save_dir, color="red"):
        # Load model and processor
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

        texts = ["doorway", "entrance", "open door", "hallway"]

        # Preprocess
        inputs = processor(text=texts, images=image, return_tensors="pt")

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Postprocess (get bounding boxes in pixel space)
        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
        results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)

        # Get boxes and scores
        boxes = results[0]["boxes"]
        scores = results[0]["scores"]
        labels = results[0]["labels"]

        # Sort by highest score
        sorted_indices = scores.argsort(descending=True)

        # Apply sorting
        boxes = boxes[sorted_indices]
        scores = scores[sorted_indices]
        labels = labels[sorted_indices]

        # Print + visualize - ONLY the first one? most confident
        for box, score, label in zip(boxes, scores, labels):
            print(f"Detected '{texts[label]}' with confidence {score:.2f} at {box.tolist()}")

            # Draw on image
            draw = image.copy()
            draw_box = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            plt.close('all')
            plt.imshow(draw)
            ax_o = plt.gca()
            ax_o.add_patch(draw_box)
            plt.title(f"{texts[label]}: {score:.2f}")
            plt.axis("off")
            plt.savefig(os.path.join(save_dir, f"owl_det_{ind}.png"))
            plt.close()

            return box, texts[label]
        
        return None, None    

    def is_detected(self, points, new_pt):
        if len(points) == 0:
            return False
        existing = np.array(points)
        distances = np.linalg.norm(existing - new_pt, axis=1)
        thresh = 2
        if np.any(distances < thresh):
            return True
        return False

    def count_room_clusters(self, regions, max_distance=5):
        count = 0
        rooms = []
        for room, pts in regions.items():
            if len(pts) <= 1:
                continue

            pts_arr = np.array(pts)
            dists = pdist(pts_arr)  # pairwise distances between all points
            if np.any(dists <= max_distance):
                count += 1
                rooms.append(room)
        return rooms

    def frontier_exp(self, cnt_step, pts_normal, cam_pose_tsdf, curr_angle, pts_pixs):
        # original frontier-based exploration from baseline methods
        self.local = False

        prompt_points_pix = []

        pts_normal, angle, pts_pix, fig, unoccupied = self.tsdf_planner.find_next_pose(
            pts=pts_normal,
            angle=curr_angle,
            flag_no_val_weight=cnt_step < self.cfg.min_random_init_steps,
            **self.cfg.planner,
        )
        # add next point in voxel space
        pts_pixs = np.vstack((pts_pixs, pts_pix))
        pts_normal = np.append(pts_normal, self.floor_height)
        pts = pos_normal_to_habitat(pts_normal)

        self.past_steps.append([pts, angle])
        return pts, angle, pts_pixs


    def door_frontiers(self, cnt_step, rgb_im, regions, room_list, rooms_to_explore, sim_scores, depth, pts_normal, curr_angle, pts_pixs, cam_pose_tsdf, camera_tilt):
        pts, angle = None, None
        curr_pts = pts_normal
        curr_vox = self.tsdf_planner.world2vox(curr_pts)
        rooms = self.count_room_clusters(regions) # Not Used?
        outside = False

        print("room list", room_list)
        print("rooms to explore", rooms_to_explore)
        relevant = False
        for r in rooms_to_explore:
            if r.lower() in room_list:
                relevant = True
                self.curr_room_goal = r
        
        if len(room_list) > 1:
            # outside room
            outside = True

        if (relevant or self.local) and self.curr_turns < 3 and len(room_list) < self.max_det_rooms:
            logging.info(f"SWITCHING TO LOCAL EXPLORATION")
            if self.local == False:
                # previously outside of local exploration
                self.prev_global_state = self.past_steps[-1]

            print(f"room: {self.curr_room_goal}")
            if outside and not self.local:
                logging.info(f"Outside relevant room: {self.curr_room_goal}")
                image = pil_to_tensor(rgb_im).to(dtype=torch.float32, device='cuda')
                heatmap = self.featurizer.text_alignment(self.curr_room_goal, image)
                segmented, centroid = self.featurizer.segment(heatmap)
                
                # Plot heatmap:
                fig_ft, (ax_ft) = plt.subplots(figsize=(15, 8))
                im = ax_ft.imshow(heatmap, cmap='viridis')  # or 'hot', 'plasma', etc.
                fig_ft.colorbar(im, ax=ax_ft)
                ax_ft.set_title("Lang-Aligned Similarity Heatmap")
                ax_ft.axis("off")
                #plt.show()
                plt.savefig(os.path.join(self.episode_data_dir, f"{cnt_step}_{self.curr_room_goal}_entrance.png"))
                plt.close(fig_ft)

                if centroid != None:
                    # TODO: add local possibilties to queue if more than one?
                    x, y, = centroid
                    world_pos = pixel2world(x, y, depth[int(y), int(x)], self.cam_intr, cam_pose_tsdf)
                    logging.info(f"world pos: {world_pos}")
                    goal_vox = self.tsdf_planner.world2vox(world_pos)
                    logging.info(f"goal vox for room: {goal_vox}")

                    # step partway towards target region
                    dir = world_pos - curr_pts
                    new_pos = curr_pts + dir * self.partial_step_ratio
                    
                    # get nearest unoccupied voxel point
                    pts_normal, nearest_vox = self.tsdf_planner.pix_to_voxel(pts_normal, new_pos)
                    logging.info(f"nearest vox: {nearest_vox}")
                    pts_pix = nearest_vox

                    # tsdf planner
                    rel_dir = nearest_vox - curr_vox[:2]
                    logging.info(f"rel dir: {rel_dir}")

                    # if next point is same as current (should not happen)
                    if rel_dir[0] == 0 and rel_dir[1] == 0:
                        # turn 180
                        angle = curr_angle + np.pi
                    else:
                        rel_dir = rel_dir / np.linalg.norm(rel_dir)
                        angle = np.arctan2(rel_dir[1], rel_dir[0])
                    #angle = np.arctan2(rel_dir[1], rel_dir[0])
                else:
                    pts_normal = None
            else:
                self.local = True

                # inside room
                logging.info(f"Inside relevant room: {self.curr_room_goal}")

                # OPTION 1: feature heatmap for anchor/target object
                #heatmap = self.featurizer.text_alignment(self.curr_room_goal, rgb_im)

                # OPTION 2: query llm to pick directions
                #self.tsdf_planner.find_prompt_points_within_view()

                # TODO: add target detection
                #bbox = self.detect_obj(cnt_step, rgb_im, target)
                #if bbox is not None:
                    # focused target viewing
                #    world_pos = pixel2world(x, y, depth[int(y), int(x)], self.cam_intr, cam_pose)
    
                rads = 90 * np.pi / 180 #self.cfg.hfov * np.pi / 180
                self.curr_turns += 1 

                pts_pix = pts_pixs[-1]
                angle = curr_angle + rads
                logging.info(f"Turning from {curr_angle} to {angle}")
        else:
            if self.local:
                self.check_stop = True
                # TODO: move back to previous point?
                if self.prev_global_state != None:
                    logging.info(f"Returning to previous position")
            else:
                self.check_stop = False
            
            # FIND OBJECT GOAL
            self.curr_turns = 0
            self.local = False
            # avoid staying in same room
            pts_normal, angle, pts_pix, island = self.tsdf_planner.find_next_pose_doorway(pts_normal, self.unexp_w, curr_angle, self.agent, camera_tilt, use_find_normal=self.use_find_normal)
        
        if pts_normal is None:
            pts_normal = curr_pts #pos_habitat_to_normal(pts)
            angle = curr_angle
            step_default = True
            logging.info("Switching back to default frontier exploration")
            return None, None, pts_pixs
        else:
            # in voxel space? all past explored points
            pts_pixs = np.vstack((pts_pixs, pts_pix))
            pts_normal = np.append(pts_normal[:2], self.floor_height)
            pts = pos_normal_to_habitat(pts_normal)
    
            self.past_steps.append([pts, angle])
            logging.info(f"Door frontier: current {curr_vox}, next {pts_pix}")

        return pts, angle, pts_pixs