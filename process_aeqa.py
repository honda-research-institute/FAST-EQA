import json
import numpy as np

file = 'data/open-eqa-v0-hm3d.json'
pos_file = 'data/aeqa_questions-184.json'
dist_file = 'data/aeqa_gt_path_length.json'

with open(file, 'r') as f:
    question_data = json.load(f)

with open(dist_file, 'r') as f:
    dist_data = json.load(f)

with open(pos_file, 'r') as f:
    pos_data = json.load(f)

scene_init = {}
new_data = []
for q in pos_data:
    scene = q["episode_history"].split('-')[-1]
    scene_init.update({scene:[q["position"], q["rotation"], q["episode_history"]]})

print(scene_init)
dist = 0
for d in question_data:
    print(d["question"])
    gt_dist = dist_data[d["question_id"]]
    scene_id = d["episode_history"].split('/')[-1]
    scene = scene_id.split('-')[-1]
    if scene not in scene_init:
        print("SKIPPING: ", scene)
        continue
    d["gt_dist"] = gt_dist
    d["position"] = scene_init[scene][0]
    d["rotation"] = scene_init[scene][1]
    d["episode_history"] = scene_init[scene][2]
    new_data.append(d)

with open('data/open-eqa-v0-hm3d-gtdist.json', 'w') as f:
    json.dump(new_data, f, indent=4)
