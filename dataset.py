import os
import csv
import json
import logging

def load_hmeqa(cfg):
    # Load dataset
    with open(cfg.question_data_path) as f:
        questions_data = [
            {k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)
        ]
    with open(cfg.init_pose_data_path) as f:
        init_pose_data = {}
        for row in csv.DictReader(f, skipinitialspace=True):
            init_pose_data[row["scene_floor"]] = {
                "init_pts": [
                    float(row["init_x"]),
                    float(row["init_y"]),
                    float(row["init_z"]),
                ],
                "init_angle": float(row["init_angle"]),
            }
    logging.info(f"Loaded {len(questions_data)} questions.")

    return questions_data, init_pose_data

def load_express(cfg):
    init_pose_data = {}

    with open(cfg.question_data_path, "r") as file:
        question_data = json.load(file)

    for d in question_data:
        d["scene"] = d["scene_id"].split("/")[-1]
        d["split"] = d["scene_id"].split("/")[1]
        #init_angle = 2 * math.acos(d["start_rotation"][0])
        init_pose_data[d["episode_id"]] = {
        "init_pts": d["start_position"], # CHANGE
        #"init_angle": init_angle,
        "rotation": d["start_rotation"]
    }

    return question_data, init_pose_data

def load_mthm3d(cfg):
    # Load dataset
    with open(cfg.question_data_path) as f:
        questions_data = [
            {k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)
        ]
    with open(cfg.init_pose_data_path) as f:
        init_pose_data = {}
        for row in csv.DictReader(f, skipinitialspace=True):
            init_pose_data[row["scene_floor"]] = {
                "init_pts": [
                    float(row["init_x"]),
                    float(row["init_y"]),
                    float(row["init_z"]),
                ],
                "init_angle": float(row["init_angle"]),
            }
    logging.info(f"Loaded {len(questions_data)} questions.")

    return questions_data, init_pose_data

# Note: load from json file provided in 3D-Mem repo
def load_openeqa(cfg):
    init_pose_data = {}

    with open(cfg.question_data_path, "r") as file:
        question_data = json.load(file)

    for d in question_data:
        d["scene"] = d["episode_history"]
        init_pose_data[d["scene"]] = {
        "init_pts": d["position"],
        "rotation": d["rotation"]
    }

    return question_data, init_pose_data