import json
import numpy as np

file = 'data/express-bench.json'

with open(file, 'r') as f:
    data = json.load(f)

dist = 0
for d in data:
    dist = 0
    steps = sorted(d["actions"].keys(), key=lambda x: int(x.split('_')[1]))
    for i in range(1, len(steps)):
        prev_pos = np.array(d["actions"][steps[i-1]]["position"])
        curr_pos = np.array(d["actions"][steps[i]]["position"])
        dist += np.linalg.norm(curr_pos[:2] - prev_pos[:2])

    print(f"Total distance moved: {dist:.4f} meters")
