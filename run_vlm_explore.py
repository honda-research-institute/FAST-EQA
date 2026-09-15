"""
Run EQA in Habitat-Sim with VLM exploration.

"""

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"

import numpy as np
np.set_printoptions(precision=3)

import re
import time
import csv
import pickle
import logging
import json
import math
import torch
import quaternion
import ast
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from scipy.spatial.distance import pdist
import habitat_sim
from transformers import CLIPProcessor, CLIPModel
from habitat_sim.utils.common import quat_to_coeffs, quat_from_coeffs, quat_from_angle_axis, quat_to_angle_axis
from habitat.utils.visualizations import maps

from src.habitat import (
    make_simple_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
    get_quaternion
)
from src.geom import get_cam_intr, get_scene_bnds, pixel2world
from src.vlm import VLM
from src.tsdf import TSDFPlanner
from explore import ExplorationAgent
from relevancy import get_clip_rel, get_clip_rel_fast
from am_radio import AMRadio
from dataset import *


def numpy_to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.generic,)):  # NumPy scalars
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): numpy_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [numpy_to_jsonable(v) for v in obj]
    return obj

def get_distance(sim, start, end):
    path = habitat_sim.ShortestPath()
    path.requested_start = start
    path.requested_end = end
    found = sim.pathfinder.find_path(path)
    if not found:
        return np.linalg.norm(np.array(end)-np.array(start))
    return path.geodesic_distance

def quaternion_to_yaw(q):
    w, x, y, z = q
    # Yaw (rotation about Y-axis)
    siny_cosp = 2.0 * (w * y + x * z)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw 

def query_stop(cfg, is_open_vocab, vlm, vlm_question, top_images, top_inds):
    early_stop = False
    gpt_stop_prompt = cfg.prompts.gpt_stop.format(vlm_question)
    logging.info(f"Query stop condition with images: {top_inds}.")
    pred_token = vlm.query_gpt(gpt_stop_prompt, top_images, multichoice=not is_open_vocab).split('.')[0]
    print("pred_token", pred_token)
    
    if pred_token == 'B' or 'stop' in pred_token:
        logging.info(f"STOPPING")
        early_stop = True
    
    return early_stop

def clean_room_name(name):
    # Remove leading "or", "and", etc.
    name = re.sub(r"^(or|and)\s+", "", name.strip(), flags=re.IGNORECASE)
    # Remove trailing punctuation
    name = re.sub(r"[^\w\s]", "", name)
    return name.strip().lower()

def is_obstructed(depth, thresh_m= 0.4, min_frac=0.7):
    # check depth values in image
    # invalid values are 0 or infinite
    valid = np.isfinite(depth) & (depth > 0)
    if valid.sum() == 0:
        return False  # no signal or all-black image

    # depth in habitat assumed to be along ray, in meters
    close = (depth < thresh_m) & valid
    close_frac = close.sum() / valid.sum()
    print(f"close frac: {close_frac}")

    # percentage of close points
    obstructed = (close_frac >= min_frac)
    logging.info(f"obstructed: {obstructed}")

    return obstructed

def main(cfg):

    # Set up camera parameters:
    camera_tilt = cfg.camera_tilt_deg * np.pi / 180
    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    # Load dataset
    if cfg.dataset == "HM-EQA":
        is_open_vocab = False
        questions_data, init_pose_data = load_hmeqa(cfg)
    elif cfg.dataset == "EXPRESS":
        is_open_vocab = True
        questions_data, init_pose_data = load_express(cfg)
    elif cfg.dataset == "MT-HM3D":
        is_open_vocab = False
        questions_data, init_pose_data = load_mthm3d(cfg)
    elif cfg.dataset == "A-EQA":
        is_open_vocab = True
        questions_data, init_pose_data = load_openeqa(cfg)
    else:
        print("Dataset not supported")

    '''
    Load VLM
        VLM class is used to query all the VL-Models used in this pipeline.
        Prismatic-VLM, GPT-4o, Mistral-small
        TODO: move CLIP functions inside it.
    '''
    vlm = VLM(cfg.vlm)

    # CLIP: Load the model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float16 if torch.cuda.is_available() else None)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    '''
    Load AM-RADIO featurizer:
    
    AM-Radio features are used to get heatmaps for doorways.
    TODO: Are there better features that can be used for this task?
    '''
    featurizer = AMRadio(adaptor_names=cfg.feat_adaptor)

    '''
    Main TODO:
    [] There are too many nested loops. Make functions for handling different states and make main() readable
    [] Clearly Explain all the Data Structures specifically pertaining to memory. Draw diagrams.
    [] Make handling of Multi-Target setting more elegent.
    [] Get all the tokens (Hugging Face, GPT, etc.) from environment flags to avoid adding them in cfg file.
    '''

    '''
    Results Logging (results_all):

    {
        'question_ind': 0, 
        'step_0': 
            {
                'pts': [1.551065, 0.13908827, -3.0018978], 
                'angle': 0.0953831335597118, 
                'smx_vlm_rel': array([0.29421497, 0.70578503]), 
                'combined_rel': 0.2230814966776955, 
                'step_time': 8.092603206634521
            },
        'step_1': {...}
    }
    '''

    # Main question loop
    cnt_data = 0
    results_all = []
    ov_scores = []
    step_times = []
    question_times = []

    # clip_50 = [ 10, 12, 33, 37, 51, 67, 87, 92, 93, 100, 102, 
    #             104, 123, 140, 142, 189, 190, 203, 226, 242, 
    #             246, 257, 304, 352, 373, 380, 381, 382, 388, 
    #             390, 393, 404, 410, 421, 425, 432, 446, 458, 
    #             461, 476, 482, 492]
    
    # sensitive_questions = [ 10, 12, 19, 27, 28, 33, 37, 43, 51, 60, 64, 67, 
    #                         76, 82, 84, 87, 92, 93, 100, 102, 104, 106, 109, 
    #                         111, 113, 116, 119, 123, 129, 137, 138, 140, 142, 
    #                         173, 174, 176, 178, 182, 183, 185, 187, 189, 190, 
    #                         195, 197, 203, 225, 226, 238, 242, 246, 252, 257, 
    #                         259, 265, 278, 280, 286, 304, 305, 309, 322, 345, 
    #                         349, 351, 352, 370, 373, 380, 381, 382, 388, 390, 
    #                         393, 402, 404, 406, 410, 421, 423, 425, 429, 432, 
    #                         435, 446, 458, 461, 467, 469, 474, 476, 481, 482, 492]

    # questions_of_interest = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]

    for question_ind in tqdm(range(cfg.start_idx, cfg.end_idx)):
    # for question_ind in questions_of_interest:

        # if question_ind not in questions_of_interest:
        #     continue

        # if question_ind in sensitive_questions:
        #     continue

        # Dynamically adjust CLIP weight
        clip_sim_w = cfg.clip_sim_w

        # if question_ind in clip_50:
        #     clip_sim_w = 0.5

        q_start = time.time()
        tot_dist = 0

        early_stop = False

        # Extract question
        question_data = questions_data[question_ind]
        scene = question_data["scene"]
        
        question = question_data["question"]
        answer = question_data["answer"]

        # format question as MCQA or open vocab
        if cfg.dataset == "HM-EQA":
            floor = question_data["floor"]
            scene_floor = scene + "_" + floor
            choices = [c for c in ast.literal_eval(question_data["choices"])]
            init_pts = init_pose_data[scene_floor]["init_pts"]
            init_angle = init_pose_data[scene_floor]["init_angle"]
            vlm_question = question
            vlm_pred_candidates = ["A", "B", "C", "D"]
            for token, choice in zip(vlm_pred_candidates, choices):
                vlm_question += "\n" + token + "." + " " + choice
        elif cfg.dataset == "EXPRESS":
            floor = '0' # placeholder
            split = question_data["split"]
            vlm_question = question
            init_pts = question_data["start_position"]
            init_rot = np.quaternion(question_data["start_rotation"][0], question_data["start_rotation"][1], question_data["start_rotation"][2], question_data["start_rotation"][3])
            
            step_length = question_data["step_length"]
            gt_dist = question_data["geodesic_distance"]
        elif cfg.dataset == "MT-HM3D":
            floor = question_data["floor"]
            scene_floor = scene + "_" + floor
            choices = [c for c in ast.literal_eval(question_data["choices"])]
            if scene_floor not in init_pose_data:
                logging.info(f"Skipping {scene_floor}, no init pose")
                continue
            init_pts = init_pose_data[scene_floor]["init_pts"]
            init_angle = init_pose_data[scene_floor]["init_angle"]
            vlm_question = question
            vlm_pred_candidates = ["A", "B", "C", "D"]
            for token, choice in zip(vlm_pred_candidates, choices):
                vlm_question += "\n" + token + "." + " " + choice
        elif cfg.dataset == "A-EQA":
            floor = '0' # placeholder
            vlm_question = question
            init_pts = init_pose_data[scene]["init_pts"]
            #init_angle = init_pose_data[scene]["init_angle"]
            init_rot = init_pose_data[scene]["rotation"]
            step_length = None
            gt_dist = question_data["gt_dist"]
        
        logging.info(f"\n========================================================")
        logging.info(f"Index: {question_ind} Scene: {scene} Floor: {floor}")
        logging.info(f"Question:\n{vlm_question}\nAnswer: {answer}")

        # Set data dir for this question - set initial data to be saved
        episode_data_dir = os.path.join(cfg.output_dir, str(question_ind))
        os.makedirs(episode_data_dir, exist_ok=True)
        result = {"question_ind": question_ind}
        result = {"scene": scene}
        result = {"question": question}
        result = {"expected_answer": answer}

        # Set up scene in Habitat
        try:
            simulator.close()
        except:
            pass
        
        if cfg.dataset == "HM-EQA" or cfg.dataset == "A-EQA":
            scene_mesh_dir = os.path.join(
                cfg.scene_data_path, scene, scene[6:] + ".basis" + ".glb"
            )
            navmesh_file = os.path.join(
                cfg.scene_data_path, scene, scene[6:] + ".basis" + ".navmesh"
            )
        elif cfg.dataset == "MT-HM3D":
            for scene_path in cfg.scene_data_path:
                if os.path.exists(os.path.join(scene_path, scene)):
                    scene_data_path = scene_path
                    break
            scene_mesh_dir = os.path.join(
                scene_data_path, scene, scene[6:] + ".basis" + ".glb"
            )
            navmesh_file = os.path.join(
                scene_data_path, scene, scene[6:] + ".basis" + ".navmesh"
            )
        elif cfg.dataset == "EXPRESS":
            scene_mesh_dir = os.path.join(
                cfg.scene_data_path, split, scene, scene[6:] + ".basis" + ".glb"
            )
            navmesh_file = os.path.join(
                cfg.scene_data_path, split, scene, scene[6:] + ".basis" + ".navmesh"
            )

        sim_settings = {
            "scene": scene_mesh_dir,
            "default_agent": 0,
            "sensor_height": cfg.camera_height,
            "width": img_width,
            "height": img_height,
            "hfov": cfg.hfov,
        }
        sim_cfg = make_simple_cfg(sim_settings)
        simulator = habitat_sim.Simulator(sim_cfg)

        pathfinder = simulator.pathfinder
        pathfinder.seed(cfg.seed)
        pathfinder.load_nav_mesh(navmesh_file)
        agent = simulator.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
        pts = init_pts
        
        # Floor - use pts height as floor height
        if cfg.dataset == "EXPRESS":
            angle = quaternion_to_yaw(question_data["start_rotation"])
            rotation = init_rot
        elif cfg.dataset == "A-EQA":
            init_quat = quaternion.quaternion(*init_rot)
            angle, axis = quat_to_angle_axis(init_quat)
            angle = angle * axis[1] / np.abs(axis[1])
            rotation = get_quaternion(angle, 0)
        else:
            angle = init_angle
            rotation = quat_to_coeffs(
                quat_from_angle_axis(angle, np.array([0, 1, 0]))
                * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
            ).tolist()

        pts_normal = pos_habitat_to_normal(pts)
        floor_height = pts_normal[-1]
        tsdf_bnds, scene_size = get_scene_bnds(pathfinder, floor_height)
        num_step = int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio)
        logging.info(
            f"Scene size: {scene_size} Floor height: {floor_height} Steps: {num_step}"
        )

        # Initialize TSDF
        tsdf_planner = TSDFPlanner(
            vol_bnds=tsdf_bnds,
            voxel_size=cfg.tsdf_grid_size,
            floor_height_offset=0,
            pts_init=pos_habitat_to_normal(pts),
            init_clearance=cfg.init_clearance * 2,
            simulator=simulator
        )

        # Initialize exploration agent
        exploration = ExplorationAgent(cfg, vlm, featurizer, tsdf_planner, agent, episode_data_dir, floor_height)
        exploration.past_steps.append([pts, angle])

        # Get Rooms and Visual Goals:
        # We need 2 data structres: rooms_to_explore and visual_goals:
        #############################

        # Extract rooms and visual goals from question using a single prompt:
        goals = vlm.extract_goals_and_rooms(question)

        # Get Rooms to Explore:
        rooms_to_explore = goals["rooms"]

        # Get Visual Goals:
        visual_goals = goals["visual_goals"]
        
        mt_clip_scores = []
        mt_combined_scores = []
        for goal in visual_goals:
            mt_clip_scores.append([])
            mt_combined_scores.append([])

        num_targets = len(visual_goals)

        logging.info(f"Rooms to explore: {rooms_to_explore}")
        logging.info(f"num targets: {num_targets}")
        logging.info(f"visual goal: {visual_goals}")

        #############################
        # rooms_to_explore = vlm.get_rooms_needed(question)
        # logging.info(f"Rooms to explore: {rooms_to_explore}")

        # task_steps = vlm.extract_goals(question)
        # logging.info(f"LLM task plan (Extracted Goals): {task_steps}")
        # if type(task_steps) != type([]):
        #     task_steps = [task_steps]

        # anchors = []  # Not Used
        # targets = []  # Not Used
        # mt_clip_scores = []
        # mt_combined_scores = []

        # visual_goals = []
        # for goal in task_steps:
        #     visual_goal = ''
        #     if "room" in goal.keys():
        #         visual_goal = goal["room"]
        #     if "anchor" in goal.keys():
        #         anchors.append(goal["anchor"])
        #         visual_goal += ' '
        #         visual_goal += goal["anchor"]
        #     if "target" in goal.keys():
        #         targets.append(goal["target"])
        #         visual_goal += ' '
        #         visual_goal += goal["target"]
            
        #     mt_clip_scores.append([])
        #     mt_combined_scores.append([])
        #     visual_goals.append(visual_goal)

        # logging.info(f"Visual Goal(s): {visual_goals}")
        # num_targets = len(task_steps)
        # logging.info(f"Num Targets: {num_targets}")

        ############################3

        # Run steps
        pts_pixs = np.empty((0, 2))  # for plotting path on the image

        '''
        Memory Data Structures:

        M = [M1, M2, M3, ..., Mn]
        
        --                                                --  
        |      Mi               |        Mj                 |
        |   image i             |    image j                |
        |   rel_score i         |    rel_score j            |
        |   clip_score i        |    clip_score j           |
        |   combined_score i    |    combined_score j       | 
        --                                                --
        '''

        images = []             # Saved images at each step
        relevancy_scores = []   # Prismatic Relavency: Given Current View, can you answer the question?
        clip_scores = []        # CLIP similarity scores: Is current view similar to visual goal?
        regions = {}            # Semantic Map: room_name -> list of voxel coords belonging to that room
        combined_scores = []    # Combined scores: weighted sum of Prismatic relevancy and CLIP similarity

        # States:
        curr_turns = 0
        curr_pos = []

        for cnt_step in range(num_step):
            step_start = time.time()
            logging.info(f"\nStep: {cnt_step} ---------------------------------------")

            # Save step info and set current pose
            step_name = f"step_{cnt_step}"

            # logging distance
            if cnt_step > 0:
                dist = get_distance(simulator, prev_pts, pts)
                tot_dist += dist
            prev_pts = pts
            
            agent_state.position = pts
            agent_state.rotation = rotation
            agent.set_state(agent_state)
            pts_normal = pos_habitat_to_normal(pts)
            result[step_name] = {"pts": pts, "angle": angle}
            curr_pos = pts_normal
            curr_angle = angle
            logging.info(f"Current pts: {curr_pos}, current angle: {angle}")
            curr_vox = tsdf_planner.world2vox(curr_pos)

            # Update camera info
            sensor = agent.get_state().sensor_states["depth_sensor"]
            quaternion_0 = sensor.rotation
            translation_0 = sensor.position
            cam_pose = np.eye(4)
            cam_pose[:3, :3] = quaternion.as_rotation_matrix(quaternion_0)
            cam_pose[:3, 3] = translation_0
            cam_pose_normal = pose_habitat_to_normal(cam_pose)
            cam_pose_tsdf = pose_normal_to_tsdf(cam_pose_normal)

            # Get observation at current pose - skip black image, meaning robot is outside the floor
            obs = simulator.get_sensor_observations()
            rgb = obs["color_sensor"] # in RGBA format
            depth = obs["depth_sensor"]
            
            
            rgb_im = Image.fromarray(rgb, mode="RGBA").convert("RGB")
            num_black_pixels = np.sum(
                np.sum(rgb_im, axis=-1) == 0
            )  # sum over channel first
            logging.info(f"# Black Pixels: {num_black_pixels}")

            black_pixel_thresh = cfg.black_pixel_ratio * img_width * img_height

            obstructed = is_obstructed(depth, cfg.depth_thresh, cfg.min_depth_ratio)
            
            # try turning if current view not valid
            max_turns = int(360 / cfg.turn_angle) - 1
            turns = 0
            while(obstructed or num_black_pixels > black_pixel_thresh) and (turns < max_turns):
                if cfg.save_obs:
                    plt.imsave(
                        os.path.join(episode_data_dir, "{}_turn_{}.png".format(cnt_step, turns)), rgb
                    )
                logging.info(f"Invalid view, turning {cfg.turn_angle}.")
                
                turn_rad = cfg.turn_angle / 180 * np.pi
                angle += turn_rad
                rotation = quat_to_coeffs(
                    quat_from_angle_axis(angle - np.pi/2, np.array([0, 1, 0]))
                    * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
                ).tolist()

                agent_state.position = pts
                agent_state.rotation = rotation
                agent.set_state(agent_state)

                curr_angle = angle
                
                # Update camera info
                sensor = agent.get_state().sensor_states["depth_sensor"]
                quaternion_0 = sensor.rotation
                translation_0 = sensor.position
                cam_pose = np.eye(4)
                cam_pose[:3, :3] = quaternion.as_rotation_matrix(quaternion_0)
                cam_pose[:3, 3] = translation_0
                cam_pose_normal = pose_habitat_to_normal(cam_pose)
                cam_pose_tsdf = pose_normal_to_tsdf(cam_pose_normal)

                # Get observation at current pose - skip black image, meaning robot is outside the floor
                obs = simulator.get_sensor_observations()
                rgb = obs["color_sensor"] # in RGBA format
                depth = obs["depth_sensor"]
                  
                rgb_im = Image.fromarray(rgb, mode="RGBA").convert("RGB")

                # recalculate black pixels
                num_black_pixels = np.sum(
                    np.sum(rgb_im, axis=-1) == 0
                )  # sum over channel first
                logging.info(f"# Black Pixels: {num_black_pixels}")

                # check obstructed again
                obstructed = is_obstructed(depth, cfg.depth_thresh, cfg.min_depth_ratio)
                
                turns += 1

            if cfg.save_obs:
                plt.imsave(
                    os.path.join(episode_data_dir, "{}.png".format(cnt_step)), rgb
                )

            if num_black_pixels < black_pixel_thresh:

                # TSDF fusion - also fuse if black image?
                exploration.tsdf_planner.integrate(
                    color_im=rgb,
                    depth_im=depth,
                    cam_intr=cam_intr,
                    cam_pose=cam_pose_tsdf,
                    obs_weight=1.0,
                    margin_h=int(cfg.margin_h_ratio * img_height),
                    margin_w=int(cfg.margin_w_ratio * img_width),
                )
                
                rgb_im = Image.fromarray(rgb, mode="RGBA").convert("RGB")
                images.append(rgb_im)
                
                # Get VLM prediction for rooms
                room = vlm.generate(cfg.prompts.room, rgb_im)
                logging.info(f"Detected room: {room}")
                room_list = [clean_room_name(r) for r in room.split(',')]
                logging.info(f"Detected room list: {room_list}")
                # NOTE: using middle pixel for now
                x = img_height / 2
                y = img_width / 2
                world_pos = pixel2world(x, y, depth[int(y), int(x)], cam_intr, cam_pose_tsdf)
                vox_coord = tsdf_planner.world2vox(world_pos)
                # empty string check 
                if len(room_list) > 0 and len(room_list) < cfg.max_det_rooms:
                    # NOTE: if multiple rooms detected, save each
                    for det_room in room_list:
                        if det_room not in regions.keys():
                            regions.update({det_room:[]})
                        regions[det_room].append(vox_coord[:2].tolist())

                # Get VLM relevancy
                prompt_rel = f"\nConsider the question: '{question}'. Are you confident about answering the question with the current view? Answer with Yes or No."
                # probability (from VLM) of yes/no token as image relevance
                smx_vlm_rel = vlm.get_loss(rgb_im, prompt_rel, ["Yes", "No"])
                logging.info(f"Relavency [Y|N] Probability: {smx_vlm_rel}")
                result[step_name]["smx_vlm_rel"] = smx_vlm_rel

                # score relevance
                sim_scores = get_clip_rel_fast(clip_model, processor, visual_goals + ["doorway to another room"], rgb_im)
                logging.info(f"CLIP scores: {sim_scores}")

                clip_scores.append(sim_scores[0][0].item())
                logging.info(f"CLIP score: {sim_scores[0][0].item()}")

                relevancy_scores.append(smx_vlm_rel[0])
                sorted_rel = np.flip(np.argsort(relevancy_scores))
                if len(sorted_rel) > 2:
                    top_k = 3
                else:
                    top_k = len(sorted_rel)

                top_images = [images[i] for i in sorted_rel[:top_k]]
                logging.info(f"Current top images based on vlm: {sorted_rel[:3]}")

                # change later to work for any number of targets
                if cfg.dataset == "MT-HM3D":
                    for i in range(num_targets):
                        mt_clip_scores[i].append(sim_scores[0][i].item())
                        logging.info(f"MT CLIP score: {sim_scores[0][i]}")
                        mt_combined_scores[i].append(clip_sim_w * sim_scores[0][i].item() + (1 - clip_sim_w) * smx_vlm_rel[0])

                    result[step_name]["combined_rel_mt"] = mt_combined_scores
                else:
                    result[step_name]["combined_rel"] = clip_sim_w * sim_scores[0][0].item() + (1 - clip_sim_w) * smx_vlm_rel[0]
                    combined_scores.append(result[step_name]["combined_rel"])
                    logging.info(f"Combined (VLM + CLIP) Score: {result[step_name]['combined_rel']}")

                step_default = False

                if cnt_step < num_step:

                    # EXPLORATION LOGIC
                    if not cfg.default_exp and cfg.exp_type == 'door-frontiers':
                        
                        # Initial In-Place Rotation:
                        if cnt_step < cfg.init_spin_steps:
                            # turn 360 for first initial steps
                            # TODO: visual detection of doors/openings out of current room
                            rads = 2 * np.pi / (cfg.init_spin_steps + 1)
                            angle += rads
                            pts_pixs = np.vstack((pts_pixs, curr_vox[:2]))

                            # check stop only if one clear room detected
                            if len(room_list) == 1 and room_list[0] in rooms_to_explore:
                                top_combined = np.flip(np.argsort(combined_scores))[:cnt_step+1]
                                top_images = [images[i] for i in top_combined]

                                # multi-target
                                if cfg.dataset == "MT-HM3D":
                                    top_combined = []
                                    top_images = []
                                    for i in range(num_targets):
                                        targ_inds = np.flip(np.argsort(mt_combined_scores[i]))[:cnt_step+1]
                                        logging.info(f"Top images for target {i}: {targ_inds}")
                                        top_combined.extend(targ_inds)
                                    
                                    top_combined = list(set(top_combined))
                                    logging.info(f"top combined: {top_combined}")
                                    
                                    top_images = [images[ind] for ind in list(set(top_combined))]

                                # check early stop during first few steps if room is relevant
                                early_stop = query_stop(cfg, is_open_vocab, vlm, vlm_question, top_images, top_combined)
                                if early_stop:
                                    step_end = time.time()
                                    logging.info(f"Step time: {step_end-step_start} seconds")
                                    step_times.append(step_end-step_start)
                                    result[step_name]["step_time"] = step_end-step_start
                                    break
                        # Normal Exploration Steps:
                        else:
                            pts, angle, pts_pixs = exploration.door_frontiers(cnt_step, rgb_im, regions, room_list, rooms_to_explore, sim_scores, depth, pts_normal, curr_angle, pts_pixs, cam_pose_tsdf, camera_tilt)
                            if pts is None:
                                # default to normal frontiers
                                exploration.use_active = False
                                pts, angle, pts_pixs = exploration.frontier_exp(cnt_step, pts_normal, cam_pose_tsdf, curr_angle, pts_pixs)
                            
                            # pts, angle, pts_pixs are not used when using local exploration
                            # as next point is directly sent from local exploration module
                            if exploration.local or exploration.check_stop: 
                                # check stop   
                                # query llm with current image or most relevant up till now.
                                top_combined = np.flip(np.argsort(combined_scores))[:cfg.top_k]
                                top_images = [images[i] for i in top_combined]
                                # multi-target
                                if cfg.dataset == "MT-HM3D":
                                    top_combined = []
                                    top_images = []
                                    for i in range(num_targets):
                                        targ_inds = np.flip(np.argsort(mt_combined_scores[i]))[:cfg.top_k]
                                        logging.info(f"Top images for target {i}: {targ_inds}")
                                        #targ_images = [images[i] for i in targ_inds]#[:cnt_step+1]
                                        top_combined.extend(targ_inds)
                                    
                                    top_combined = list(set(top_combined))
                                    logging.info(f"top combined: {top_combined}")
                                    top_images = [images[ind] for ind in list(set(top_combined))]

                                # include current im - only if room facing right way??
                                overlap = list(set(room_list).intersection(set(rooms_to_explore)))
                                logging.info(f"Overlapping rooms: {overlap}")

                                if len(overlap) > 0:
                                    # only if still facing relevant room
                                    # include current image
                                    if not cnt_step in top_combined:
                                        top_images = top_images + [rgb_im]
                                        top_combined = np.append(top_combined[:cfg.top_k], cnt_step)

                                    # Limit the number of times we call query_stop() to avoid too many LLM calls:

                                    # check early stop if room is relevant
                                    early_stop = query_stop(cfg, is_open_vocab, vlm, vlm_question, top_images, top_combined)
                                    if early_stop:
                                        step_end = time.time()
                                        logging.info(f"Step time: {step_end-step_start} seconds")
                                        step_times.append(step_end-step_start)
                                        result[step_name]["step_time"] = step_end-step_start
                                        break

                    # Default Frontier Based Exploration:
                    if cfg.default_exp or step_default: # or cnt_step < 3:
                        # default frontier-based exploration
                        # Get frontier candidates
                        # pts_pixs = np.vstack((pts_pixs, curr_vox[:2]))
                        pts, angle, pts_pixs = exploration.frontier_exp(cnt_step, pts_normal, cam_pose_tsdf, curr_angle, pts_pixs)

                        #strictly inside room
                        if len(room_list) == 1 and room_list[0] in rooms_to_explore:
                            top_combined = np.flip(np.argsort(combined_scores))[:cnt_step+1]
                            top_images = [images[i] for i in top_combined]#[:cnt_step+1]
                            #top_images.append(rgb_im)
                            if cfg.dataset == "MT-HM3D":
                                top_combined = []
                                top_images = []
                                for i in range(num_targets):
                                    targ_inds = np.flip(np.argsort(mt_combined_scores[i]))[:cfg.top_k]
                                    #targ_images = [images[i] for i in targ_inds]#[:cnt_step+1]
                                    logging.info(f"Top images for target {i}: {targ_inds}")
                                    top_combined.extend(targ_inds)
                                
                                top_combined = list(set(top_combined))
                                logging.info(f"top combined: {top_combined}")
                                top_images = [images[ind] for ind in list(set(top_combined))]
                        
                            # check early stop
                            early_stop = query_stop(cfg, is_open_vocab, vlm, vlm_question, top_images, top_combined)
                            if early_stop:
                                step_end = time.time()
                                logging.info(f"Step time: {step_end-step_start} seconds")
                                step_times.append(step_end-step_start)
                                result[step_name]["step_time"] = step_end-step_start
                                break

            else:
                # should rarely occur after turning
                logging.info("Skipping black image!")
                # pts_pixs = np.vstack((pts_pixs, curr_vox[:2]))

                result[step_name]["smx_vlm_rel"] = np.array([0.01, 0.99])
                images.append(np.zeros((img_height, img_width, 3), dtype=np.uint8))
                
                # zero relevance score
                sim_score = 0.0
                clip_scores.append(sim_score)
                relevancy_scores.append(result[step_name]["smx_vlm_rel"][0])
            
                result[step_name]["combined_rel"] = sim_score + result[step_name]["smx_vlm_rel"][0]
                combined_scores.append(result[step_name]["combined_rel"])

                for i in range(num_targets):
                    mt_clip_scores[i].append(result[step_name]["smx_vlm_rel"][0])
                    mt_combined_scores[i].append(result[step_name]["smx_vlm_rel"][0])

                # get next point with default frontier exploration
                pts, angle, pts_pixs = exploration.frontier_exp(cnt_step, pts_normal, cam_pose_tsdf, curr_angle, pts_pixs)

            #############################################################################################
            # Plots: Plot Map, Path, Doorways, Regions
            #############################################################################################
            fig, unoccupied = tsdf_planner.plot(curr_pos)
            
            # convert from voxel coord angle to habitat angle with offset
            rotation = quat_to_coeffs(
                quat_from_angle_axis(angle - np.pi/2, np.array([0, 1, 0]))
                * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
            ).tolist()

            pts_pix = pts_pixs[-1]
            ax1 = fig.axes[0]
            
            ax1.scatter(curr_vox[1], curr_vox[0], c='red', s=30, label='curr')
            ax1.quiver(
                curr_vox[1],
                curr_vox[0],
                math.sin(curr_angle),
                math.cos(curr_angle),
                color="red",
                scale=5,
                angles="xy",
                alpha=0.2,
            )
            ax1.scatter(pts_pix[1], pts_pix[0], c='black', s=30, label='next')
            ax1.quiver(
                pts_pix[1],
                pts_pix[0],
                math.sin(angle),
                math.cos(angle),
                color="black",
                scale=5,
                angles="xy",
                alpha=0.2,
            )

            # Add path to ax5
            ax3 = fig.axes[2]
            ax3.plot(pts_pixs[:, 1], pts_pixs[:, 0], linewidth=5, color="black")
            ax3.scatter(pts_pixs[0, 1], pts_pixs[0, 0], c="white", s=50)
            fig.tight_layout()
            plt.savefig(
                os.path.join(episode_data_dir, "{}_map.png".format(cnt_step))
            )
            plt.close()

            #############################################################################################
            # Plot detected doorways from TSDF
            #############################################################################################
            tsdf_doorways, center_voxels, scores, island = tsdf_planner.detect_doorways_from_tsdf(pts_normal, cfg.planner.unexp_w)

            # plot current detected doorways
            if len(tsdf_doorways) > 0 and cnt_step % 5 == 0:
                fig_t, ax_t = plt.subplots()
                ax_t.imshow(island) #unoccupied)
                doorways = np.array(tsdf_doorways)
                ax_t.scatter(doorways[:, 1], doorways[:, 0], color='white', label='doorway')
                if center_voxels.ndim > 1:
                    sc = ax_t.scatter(center_voxels[:, 1], center_voxels[:, 0], c=scores, cmap='viridis', label='centers')
                    cbar = plt.colorbar(sc)
                    cbar.set_label('Cluster Score', fontsize=12)
                ax_t.legend()
                plt.savefig(os.path.join(episode_data_dir, f"{cnt_step}_tsdf_doorways.png"))
                plt.close()
            
            # plot current region map
            if len(list(regions.keys())) > 0 and cnt_step % 5 == 0:
                fig_g1, ax_g1 = plt.subplots()
                ax_g1.imshow(unoccupied)
                colors = cm.get_cmap('tab10', len(regions))
                for idx, (region, coords) in enumerate(regions.items()):
                    coords = np.array(coords)
                    #print("coords", coords)
                    ax_g1.scatter(coords[:, 1], coords[:, 0], color=colors(idx), label=region)
                ax_g1.legend(
                    bbox_to_anchor=(1.05, 1), 
                    loc='upper left',
                    borderaxespad=0.0
                )
                plt.title("Regions")
                plt.savefig(os.path.join(episode_data_dir, f"{cnt_step}_regions.png"))
                plt.close()
        
            step_end = time.time()
            logging.info(f"Step time: {step_end-step_start} seconds")
            step_times.append(step_end-step_start)
            result[step_name]["step_time"] = step_end-step_start


        #############################################################################################
        # End of Steps - Process results and query final answer from VLM. 
        # Check for Success. Utilizes top k images based on combined score.
        #############################################################################################

        top_clip_scored = np.flip(np.argsort(clip_scores))[:3]
        logging.info(f"Top images by CLIP sim: {top_clip_scored}")

        relevancy_ord = np.flip(np.argsort(relevancy_scores))

        # Check if success using weighted prediction

        # For MCQA, we use VLM + CLIP combined score to get the final answer:
        if not is_open_vocab:
            # query GPT for final answer
            if not early_stop:
                # Get most relevant images
                top_combined = np.flip(np.argsort(combined_scores))[:cfg.top_k]
                top_images = [images[i] for i in top_combined]

                if cfg.dataset == "MT-HM3D":
                    top_combined = []
                    top_images = []
                    for i in range(num_targets):
                        targ_inds = np.flip(np.argsort(mt_combined_scores[i]))[:cfg.top_k]
                        #targ_images = [images[i] for i in targ_inds]#[:cnt_step+1]
                        top_combined.extend(targ_inds)
                        logging.info(f"Top images for target {i}: {targ_inds}")
                    
                    top_combined = list(set(top_combined))
                    logging.info(f"top combined: {top_combined}")
                    top_images = [images[ind] for ind in set(top_combined)]
            
                    logging.info(f"MT combined scores: {mt_combined_scores}")
            
            # How is this being used and why is it not inside MT-HM3D block above?
            for i in range(num_targets):
                clip_rels = mt_clip_scores[i]
                top_clip_i = np.flip(np.argsort(clip_rels))[:3]
                logging.info(f"Top CLIP images for target {i} : {top_clip_i}")

            logging.info(f"Query final answer with images: {top_combined}.")
            gpt_prompt = cfg.prompts.gpt_question.format(vlm_question)
            
            print("\nFINAL PROMPT: ", gpt_prompt)
            pred_token = vlm.query_gpt(gpt_prompt, top_images, multichoice=True)
            
            success_max = pred_token == answer

        # For Open-Vocab, we use VLM score to get final answer:
        else:
            if not early_stop:
                # Get most relevant images
                top_combined = np.flip(np.argsort(combined_scores))[:cfg.top_k]
                top_images = [images[i] for i in top_combined]

            logging.info(f"Query final answer with images: {top_combined}.")
            gpt_prompt = cfg.prompts.gpt_question.format(vlm_question)

            # Final GPT Prompt:
            pred_answer = vlm.query_gpt(gpt_prompt, top_images, multichoice=False)
                        
            # Get final answer from VLM
            if "extra_answers" in question_data:
                score = vlm.score_answer_multi(question, answer, question_data["extra_answers"], pred_answer)
            else:
                score = vlm.score_answer(question, answer, pred_answer)
            #logging.info(f"Score: {score}")
            ov_scores.append(score)

        #############################################################################################
        # End of Episode:
        #############################################################################################

        # Episode summary
        logging.info(f"\n== Episode Summary")
        logging.info(f"Index: {question_ind}")
        logging.info(f"Scene: {scene}, Floor: {floor}")
        logging.info(f"Question:\n{vlm_question}\nAnswer: {answer}")
        
        if not is_open_vocab:
            logging.info(f"Predicted: {pred_token}")
            logging.info(f"Success (max): {success_max}")
            
            # Log Answer:
            result["answer"] = success_max
        else:
            logging.info(f"VLM Answer: {pred_answer}")
            if step_length is None:
                step_length = num_step
            logging.info(f"Total dist: {tot_dist}")
            E_step = (score / 5) * (step_length/max(cnt_step+1, step_length)) # from EXPRESS-Bench paper Eqn 4
            E_dist = (score / 5) * (gt_dist/max(tot_dist, gt_dist))

            LLM_Match = (score - 1) / 4 # from Open-EQA paper Eqn 1
            Eff_step = LLM_Match * (step_length/max(cnt_step+1, step_length)) # from Open-EQA paper Eqn 2
            Eff_dist = LLM_Match * (gt_dist/max(tot_dist, gt_dist))
            
            # Log Answer:
            result["answer"] = LLM_Match
            
            logging.info(f"Score: {score}")
            logging.info(f"LLM Match: {LLM_Match}")
            logging.info(f"Steps GT: {cnt_step+1}/{step_length}")
            logging.info(f"E_step: {E_step}")
            logging.info(f"E_dist: {E_dist}")
            logging.info(f"Eff_step: {Eff_step}")
            logging.info(f"Eff_dist: {Eff_dist}")
        logging.info(f"Curr pos: {curr_pos}")
        logging.info(f"Steps: {cnt_step+1}/{num_step}")
        logging.info(f"Path length: {(cnt_step+1)/num_step}")
        if cnt_step > 3:
            logging.info(
                f"Top 3 steps with highest VLM relevancy: {relevancy_ord[:3]} {[relevancy_scores[i] for i in relevancy_ord[:3]]}"
            )
        logging.info(
            f"Top 3 steps with highest CLIP relevancy: {top_clip_scored} {[clip_scores[i] for i in top_clip_scored]}"
        )

        logging.info(
            f"Top 3 steps with highest combined relevancy: {top_combined}" #{[combined_scores[i] for i in top_combined]}"
        )

        q_end = time.time()
        question_times.append(q_end-q_start)
        logging.info(f"Question time: {q_end-q_start} seconds")

        # Save data
        results_all.append(result)
        cnt_data += 1
        if cnt_data % cfg.save_freq == 0:
            with open(
                os.path.join(cfg.output_dir, f"results_{cnt_data}.pkl"), "wb"
            ) as f:
                pickle.dump(results_all, f)

    # Save all data again
    with open(os.path.join(cfg.output_dir, "results.pkl"), "wb") as f:
        pickle.dump(results_all, f)

    json_compatible_results = numpy_to_jsonable(results_all)

    with open(os.path.join(cfg.output_dir, "results.json"), "w") as f:
        json.dump(json_compatible_results, f)

    logging.info(f"\n== All Summary")
    logging.info(f"Number of data collected: {cnt_data}")
    if is_open_vocab:
        logging.info(f"Average score: {sum(ov_scores)/len(ov_scores)}")

    logging.info(f"Average step time: {sum(step_times)/len(step_times)}")
    logging.info(f"Average question time: {sum(question_times)/len(question_times)}")


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    # get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="", type=str)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    # Set up logging
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
    logging_path = os.path.join(cfg.output_dir, f"log{cfg.start_idx}_{cfg.end_idx}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
        force=True
    )

    # No Open AI | Mistral Info Logging:
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # run
    logging.info(f"***** Running {cfg.exp_name} *****")
    main(cfg)