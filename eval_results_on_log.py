# Modified from MemoryEQA repo
# Haochen Zhang, 2025

import os
import csv
import json
import re
import numpy as np
from argparse import ArgumentParser

def get_questions(dataset, data_path):
    question_dict = {}
    if dataset == 'HM-EQA':
        with open(data_path, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                #print(row)
                row["type"] = row["label"]
                question_dict.update({row['question']:row})
            return question_dict
    elif dataset == "EXPRESS":
        with open(data_path, "r") as file:
            question_data = json.load(file)
            question_dict = {d['question'].strip():d for d in question_data}
            return question_dict
    elif dataset == "MT-HM3D":
        with open(data_path, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                #print(row)
                row["type"] = row["label"]
                question_dict.update({row['question'].strip():row})
            return question_dict
    elif dataset == "A-EQA":
        with open(data_path, "r") as file:
            question_data = json.load(file)
            question_dict = {d['question'].strip():d for d in question_data}
            for key, val in question_dict.items():
                question_dict[key]["type"] = question_dict[key]["category"]
                question_dict[key]["geodesic_distance"] = question_dict[key]["gt_dist"]
            return question_dict

def find_lines_with_prefix(file_path, prefixes):
    # find matching lines
    matching_lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if any(line.startswith(prefix) for prefix in prefixes):
                matching_lines.append(line.strip())
            if line.startswith("Index: "):
                for _ in range(8):
                    matching_lines.append(next(file).strip())
            if line.startswith("== step:"):
                for _ in range(3):
                    matching_lines.append(next(file).strip())
    return matching_lines

def export_answer_comparison(file_path, question_dict):
    output_path = os.path.splitext(file_path)[0] + "_answer_compare.txt"
    lines_out = ["index\tanswer_on_log\tcorrect_answer"]

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip("\n") for line in f]

    i = 0
    while i < len(lines):
        if not lines[i].startswith("== Episode Summary"):
            i += 1
            continue

        idx = None
        question = None
        answer_on_log = None
        answer_in_log = None
        j = i + 1

        while j < len(lines):
            curr = lines[j].strip()
            if curr.startswith("========================================================"):
                break
            if curr.startswith("Question time:"):
                break

            if curr.startswith("Index:"):
                match = re.search(r"Index:\s*(\d+)", curr)
                if match:
                    idx = int(match.group(1))
            elif curr == "Question:" and j + 1 < len(lines):
                question = lines[j + 1].strip()
            elif curr.startswith("Predicted:"):
                answer_on_log = curr.split(":", 1)[1].strip()
            elif curr.startswith("Answer:"):
                answer_in_log = curr.split(":", 1)[1].strip()

            j += 1

        if answer_on_log is None:
            answer_on_log = answer_in_log if answer_in_log is not None else "N/A"

        if question in question_dict and "answer" in question_dict[question]:
            correct_answer = question_dict[question]["answer"]
        else:
            correct_answer = answer_in_log if answer_in_log is not None else "N/A"

        if idx is not None:
            lines_out.append(f"{idx}\t{answer_on_log}\t{correct_answer}")

        i = j + 1

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines_out) + "\n")

    print(f"Saved answer comparison to {output_path}")
    return output_path

def multi_file_evaluation(files_path: list, prefixes: list, args):
    # takes list of log files for evaluation
    question_dict = get_questions(args.dataset, args.data_path)
    print(question_dict)

    matching_lines = []
    for file_path in files_path:
        export_answer_comparison(file_path, question_dict)
        matching_lines.extend(find_lines_with_prefix(file_path, prefixes))

    # print(matching_lines)
    results = []
    result_dict = {}
    success_count = 0
    succ_max_count = 0
    succ_weight_count = 0
    scores = []
    step_times = []
    question_times = []
    E_step_scores = []
    E_dist_scores = []
    Eff_step_scores = []
    Eff_dist_scores = []
    LLM_match_scores = []
    score_freq = {}
    step_pts = []
    for idx, line in enumerate(matching_lines, 1):
        log = line.split(' ')
        print(line)
        if line.startswith("Index:") and len(log) > 2:
            result_dict["index"] = int(log[1])
            result_dict["scene"] = log[3]
            result_dict["floor"] = log[5]
            result_dict["question"] = matching_lines[idx + 1]
            print("q ", result_dict["question"])
            q_type = question_dict[result_dict["question"]]["type"]
            #gt_dist = question_dict[result_dict["question"]]["geodesic_distance"]
            i = 0
            while True:
                i += 1
                if matching_lines[idx + i].startswith("Scene size:"):
                    result_dict["max_step"] = int(matching_lines[idx + i].split(' ')[-1])
                    break
        elif line.startswith("== step:"):
            step = int(log[-1]) + 1
            result_dict["last_step"] = step
            step_response = matching_lines[idx + 2].split(' ')[-1]
            if step_response == "True" and "norm_early_success_step" not in result_dict.keys():
                result_dict["norm_early_success_step"] = step / result_dict.get("max_step")
        elif line.startswith("Current pts:"):
            match = re.search(r"\[([^\]]+)\]", line)
            if match:
                pts_str = match.group(1)  # "-0.655  1.087  2.993"
                pts_list = [float(x) for x in pts_str.split()]
                pts_array = np.array(pts_list)
                step_pts.append(pts_array)
        elif line.startswith("Success"):
            if '(weighted):'in log:
                result_dict["is_success_weight"] = True if log[-1] == "True" else False

            if '(max):' in log:
                result_dict["is_success_max"] = True if log[-1] == "True" else False
                last_step = result_dict.get("last_step")
                max_step = result_dict.get("max_step")
                if last_step is not None and max_step:
                    result_dict["norm_success_step"] = last_step / max_step
                else:
                    result_dict["norm_success_step"] = None
                if log[-1] == "False":
                    if last_step is not None and max_step is not None:
                        result_dict["early_fail"] = True if last_step < max_step else False
                    else:
                        result_dict["early_fail"] = False
                result_dict["type"] = q_type

        elif line.startswith("Score"):
            score = int(log[-1])
            if score not in score_freq:
                score_freq[score] = 1
            else:
                score_freq[score] += 1
        
            scores.append(score)
            result_dict["score"] = score
            result_dict["type"] = q_type
            
            step_pts = []
        elif line.startswith("LLM Match"):
            LLM_match_scores.append(float(log[-1]))
        elif line.startswith("E_step"):
            E_step_scores.append(float(log[-1]))
        elif line.startswith("E_dist"):
            E_dist_scores.append(float(log[-1]))
        elif line.startswith("Eff_step"):
            Eff_step_scores.append(float(log[-1]))
        elif line.startswith("Eff_dist"):
            Eff_dist_scores.append(float(log[-1]))
        elif line.startswith("Step time"):
            step_time = float(log[-2])
            step_times.append(step_time)
        elif line.startswith("Question time"):
            print(line)
            q_time = float(log[-2])
            question_times.append(q_time)
            
            if result_dict != {}:
                results.append(result_dict)
                result_dict = {}
            
            
    results_num = len(results)
    if not args.is_open_answer:
        norm_steps = 0
        norm_early_steps = 0
        early_count = 0
        early_fail = 0
        type_success = {}
        type_totals = {}
        for result in results:
            print(result)
            q_type = result.get("type")
            if q_type not in type_totals:
                type_totals[q_type] = 1
            else:
                type_totals[q_type] += 1

            if result.get("is_success_weight"):
                succ_weight_count += 1
            if result.get("is_success_max"):
                succ_max_count += 1
            if result.get("early_fail"):
                early_fail += 1
            if result.get("is_success_max"):
                success_count += 1
                if result.get("norm_success_step") is not None:
                    norm_steps += result.get("norm_success_step")
                if q_type not in type_success:
                    type_success[q_type] = 1
                else:
                    type_success[q_type] += 1
            if result.get("norm_early_success_step"):
                norm_early_steps += result.get("norm_early_success_step")
                early_count += 1
        
        total_fail = results_num - success_count
        print(f"Total: {results_num}, Success: {success_count}")
        print(f"Total success rate: {success_count/results_num:.2%}" if results_num else "Total success rate: N/A")
        print(f"Success (max) rate: {succ_max_count/results_num:.2%}" if results_num else "Success (max) rate: N/A")
        print(f"Success (weighted) rate: {succ_weight_count/results_num:.2%}" if results_num else "Success (weighted) rate: N/A")
        print(f"Average norm steps for success: {norm_steps/success_count:.2}" if success_count else "Average norm steps for success: N/A")
        print(f"Early success: {early_count}")
        print(f"Early failure: {early_fail}/{total_fail}")
        for key, val in type_totals.items():
            print(f"{key} success rate: {type_success.get(key, 0)}/{val}")
    else:
        type_scores = {}
        for result in results:
            print(result)
            q_type = result.get("type")
            if q_type not in type_scores:
                type_scores[q_type] = [result.get("score")]
            else:
                type_scores[q_type].append(result.get("score"))
        
        print(len(scores))
        print(results_num)
        print(f"Total: {results_num}, Success: {success_count}")
        print(f"Average score: {sum(scores)/results_num}")
        print(f"Average LLM Match: {sum(LLM_match_scores)*100/len(LLM_match_scores)}")
        print(f"Num above score 3: {sum(1 for item in scores if item >= 3)}")
        print(f"E_step (Express) score: {sum(E_step_scores)/len(E_step_scores)}")
        print(f"E_dist (Express) score: {sum(E_dist_scores)/len(E_dist_scores)}")
        print(f"Eff_step (OpenEQA) score: {sum(Eff_step_scores)/len(Eff_step_scores)}")
        print(f"Eff_dist (OpenEQA) score: {sum(Eff_dist_scores)/len(Eff_dist_scores)}")
        for key, val in type_scores.items():
            print(f"{key} average score: {sum(type_scores[key])/len(type_scores[key])}")

    if len(step_times) > 0:
        print(f"Average step time: {sum(step_times)/len(step_times)}")
        print(f"Average question time: {sum(question_times)/len(question_times)}")

    return matching_lines

if __name__ == '__main__':

    parser = ArgumentParser(description="Example command-line parser")

    # Add arguments
    parser.add_argument("--dataset", help="name of dataset", default="HM-EQA")
    parser.add_argument("--data_path", help="path to question file", default="data/questions.csv")
    parser.add_argument("--is_open_answer", help="flag for if dataset is open vocab", action='store_true')

    # Parse the arguments
    args = parser.parse_args()
    print(args)

    files_path = [
        '/data/01/hzhang/HM_EQA/vlm_hmeqa_var_2/log0_500.log'
    ]

    if not args.is_open_answer:
        prefixes = ['Index: ', 
                    '== step:', 
                    'Success (max):',
                    'Step time:',
                    'Question time:'
                    ]
    else:
        prefixes = [
            'Index: ', 
            '== step:',
            'Current pts:'
            'VLM Answer:',
            'Score:',
            'LLM Match',
            'E_step',
            'E_dist',
            'Eff_step',
            'Eff_dist',
            'Step time:',
            'Question time:'
        ]
    #matching_lines = multi_file_evaluation(files_path, prefixes)
    matching_lines = multi_file_evaluation(files_path, prefixes, args)
