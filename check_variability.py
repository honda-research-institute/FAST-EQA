import argparse
import os
import re
from collections import Counter


def parse_answer_compare_file(file_path):
    records = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, raw in enumerate(f):
            line = raw.strip()
            if not line:
                continue
            if line_idx == 0 and line.lower().startswith("index"):
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                parts = line.split()
            if len(parts) < 3:
                continue

            try:
                q_idx = int(parts[0])
            except ValueError:
                continue

            answer_on_log = parts[1].strip()
            correct_answer = parts[2].strip()
            records[q_idx] = {
                "answer_on_log": answer_on_log,
                "correct_answer": correct_answer,
            }
    return records


def get_run_name(file_path):
    base = os.path.basename(file_path).replace("_answer_compare.txt", "")
    parent = os.path.basename(os.path.dirname(file_path))
    if parent:
        return f"{parent}/{base}"
    return base


def get_log_path_from_answer_compare(file_path):
    if file_path.endswith("_answer_compare.txt"):
        return file_path.replace("_answer_compare.txt", ".log")
    return ""


def parse_log_metadata(log_path):
    metadata = {}
    if not log_path or not os.path.isfile(log_path):
        return metadata

    with open(log_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    current_final_images = ""
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("Query final answer with images:"):
            # Example: Query final answer with images: [32 30 31].
            m = re.search(r"\[([^\]]+)\]", line)
            current_final_images = m.group(1).strip() if m else ""

        if line.startswith("== Episode Summary"):
            idx = None
            question = ""
            options = []
            j = i + 1
            while j < len(lines):
                curr = lines[j].strip()

                if curr.startswith("Index:"):
                    m = re.search(r"Index:\s*(\d+)", curr)
                    if m:
                        idx = int(m.group(1))
                elif curr == "Question:" and j + 1 < len(lines):
                    question = lines[j + 1].strip()
                elif re.match(r"^[A-D]\.\s", curr):
                    options.append(curr)
                elif curr.startswith("Answer:"):
                    break
                elif curr.startswith("========================================================"):
                    break
                j += 1

            if idx is not None:
                metadata[idx] = {
                    "question": question,
                    "options": " | ".join(options),
                    "final_image_indices": current_final_images,
                }

            i = j
        i += 1

    return metadata


def normalize_image_indices(indices_text):
    if not indices_text:
        return ""
    parts = indices_text.split()
    return " ".join(parts)


def compute_variability(run_data):
    run_names = sorted(run_data.keys())
    shared_indices = None
    for run_name in run_names:
        idx_set = set(run_data[run_name].keys())
        shared_indices = idx_set if shared_indices is None else shared_indices & idx_set
    shared_indices = sorted(shared_indices) if shared_indices is not None else []

    question_rows = []
    variable_rows = []
    variable_counts_by_run = {run_name: 0 for run_name in run_names}

    for idx in shared_indices:
        answers_by_run = {}
        correct_answers = []
        for run_name in run_names:
            pred = run_data[run_name][idx]["answer_on_log"]
            corr = run_data[run_name][idx]["correct_answer"]
            answers_by_run[run_name] = pred
            correct_answers.append(corr)

        unique_answers = sorted(set(answers_by_run.values()))
        is_variable = len(unique_answers) > 1
        correct_answer = Counter(correct_answers).most_common(1)[0][0]

        row = {
            "index": idx,
            "num_unique_answers": len(unique_answers),
            "unique_answers": "|".join(unique_answers),
            "is_variable": is_variable,
            "correct_answer": correct_answer,
        }
        row.update(answers_by_run)
        question_rows.append(row)

        if is_variable:
            variable_rows.append(row)
            majority_answer = Counter(answers_by_run.values()).most_common(1)[0][0]
            for run_name in run_names:
                if answers_by_run[run_name] != majority_answer:
                    variable_counts_by_run[run_name] += 1

    run_rows = []
    total_shared = len(shared_indices)
    for run_name in run_names:
        correct_n = sum(
            1
            for idx in shared_indices
            if run_data[run_name][idx]["answer_on_log"]
            == run_data[run_name][idx]["correct_answer"]
        )
        run_rows.append(
            {
                "run": run_name,
                "num_questions_shared": total_shared,
                "accuracy_on_shared": (correct_n / total_shared) if total_shared else 0.0,
                "variable_disagree_count": variable_counts_by_run[run_name],
            }
        )

    return run_names, question_rows, variable_rows, run_rows, total_shared


def write_tsv(file_path, header, rows):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(col, "")) for col in header) + "\n")


def compute_variable_same_final_images(run_data, run_names, run_meta):
    shared_indices = None
    for run_name in run_names:
        idx_set = set(run_data[run_name].keys())
        shared_indices = idx_set if shared_indices is None else shared_indices & idx_set
    shared_indices = sorted(shared_indices) if shared_indices is not None else []

    variable_q = 0
    same_final_images_count = 0
    for idx in shared_indices:
        answers = [run_data[run_name][idx]["answer_on_log"] for run_name in run_names]
        if len(set(answers)) <= 1:
            continue
        variable_q += 1

        final_images = []
        for run_name in run_names:
            imgs = run_meta.get(run_name, {}).get(idx, {}).get("final_image_indices", "")
            final_images.append(normalize_image_indices(str(imgs)))
        if len(set(final_images)) == 1:
            same_final_images_count += 1

    return variable_q, same_final_images_count


def write_summary(
    file_path,
    total_shared,
    variable_rows,
    run_rows,
    variable_q_for_images,
    same_final_images_count,
):
    variable_q = len(variable_rows)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Total questions compared in all runs: {total_shared}\n")
        f.write(f"Variable questions: {variable_q}\n")
        f.write(
            f"Variability rate: {(variable_q / total_shared) if total_shared else 0.0:.4f}\n"
        )
        f.write(
            "Variable questions with same final image indices: "
            f"{same_final_images_count}/{variable_q_for_images}\n"
        )
        f.write(
            "Percent of variable questions with same final image indices: "
            f"{(same_final_images_count / variable_q_for_images * 100) if variable_q_for_images else 0.0:.2f}%\n"
        )
        f.write("\nPer-run stats on shared questions:\n")
        for row in run_rows:
            f.write(
                f"{row['run']}: "
                f"accuracy={row['accuracy_on_shared']:.4f}, "
                f"variable_disagree_count={row['variable_disagree_count']}, "
                f"num_questions_shared={row['num_questions_shared']}\n"
            )


def compute_correctness_transitions(run_data, run_names, run_meta):
    if len(run_names) != 2:
        raise ValueError("Correctness transition lists require exactly 2 runs in files_path.")

    run1, run2 = run_names[0], run_names[1]
    shared_indices = sorted(set(run_data[run1].keys()) & set(run_data[run2].keys()))
    false_to_true = []
    true_to_false = []

    for idx in shared_indices:
        r1_pred = run_data[run1][idx]["answer_on_log"]
        r2_pred = run_data[run2][idx]["answer_on_log"]
        correct_answer = run_data[run1][idx]["correct_answer"]
        r1_correct = (r1_pred == correct_answer)
        r2_correct = (r2_pred == correct_answer)
        r1_final_image_indices = run_meta.get(run1, {}).get(idx, {}).get(
            "final_image_indices", ""
        )
        r2_final_image_indices = run_meta.get(run2, {}).get(idx, {}).get(
            "final_image_indices", ""
        )
        final_image_indices_mismatch = (
            normalize_image_indices(r1_final_image_indices)
            != normalize_image_indices(r2_final_image_indices)
        )

        row = {
            "index": idx,
            "correct_answer": correct_answer,
            "question": run_meta.get(run1, {}).get(idx, {}).get("question", ""),
            "options": run_meta.get(run1, {}).get(idx, {}).get("options", ""),
            "run1_name": run1,
            "run1_answer": r1_pred,
            "run1_correct": r1_correct,
            "run1_final_image_indices": r1_final_image_indices,
            "run2_name": run2,
            "run2_answer": r2_pred,
            "run2_correct": r2_correct,
            "run2_final_image_indices": r2_final_image_indices,
            "final_image_indices_mismatch": final_image_indices_mismatch,
        }

        if (not r1_correct) and r2_correct:
            false_to_true.append(row)
        elif r1_correct and (not r2_correct):
            true_to_false.append(row)

    return false_to_true, true_to_false


def main():
    files_path = [
        "/data/01/hzhang/HM_EQA/vlm_hmeqa_var_1/log0_500_answer_compare.txt",
        "/data/01/hzhang/HM_EQA/vlm_hmeqa_var_2/log0_500_answer_compare.txt",
    ]

    parser = argparse.ArgumentParser(
        description="Check answer variability across multiple *_answer_compare.txt files."
    )
    parser.add_argument(
        "--output_dir",
        default="./results",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--output_prefix",
        default="variability",
        help="Prefix for output files.",
    )
    args = parser.parse_args()

    files = [p for p in files_path if os.path.isfile(p)]
    if not files:
        raise FileNotFoundError(
            "No input files found. Please edit files_path in check_variability.py."
        )

    run_data = {}
    run_names_in_order = []
    run_meta = {}
    for p in files:
        run_name = get_run_name(p)
        run_data[run_name] = parse_answer_compare_file(p)
        run_meta[run_name] = parse_log_metadata(get_log_path_from_answer_compare(p))
        run_names_in_order.append(run_name)

    (
        run_names,
        question_rows,
        variable_rows,
        run_rows,
        total_shared,
    ) = compute_variability(run_data)

    os.makedirs(args.output_dir, exist_ok=True)

    q_path = os.path.join(args.output_dir, f"{args.output_prefix}_by_question.tsv")
    v_path = os.path.join(args.output_dir, f"{args.output_prefix}_variable_questions.tsv")
    r_path = os.path.join(args.output_dir, f"{args.output_prefix}_by_run.tsv")
    s_path = os.path.join(args.output_dir, f"{args.output_prefix}_summary.txt")

    question_header = [
        "index",
        "num_unique_answers",
        "unique_answers",
        "is_variable",
        "correct_answer",
    ] + run_names
    write_tsv(
        q_path,
        question_header,
        question_rows,
    )
    write_tsv(v_path, question_header, variable_rows)
    write_tsv(
        r_path,
        [
            "run",
            "num_questions_shared",
            "accuracy_on_shared",
            "variable_disagree_count",
        ],
        run_rows,
    )
    variable_q_for_images, same_final_images_count = compute_variable_same_final_images(
        run_data, run_names_in_order, run_meta
    )
    write_summary(
        s_path,
        total_shared,
        variable_rows,
        run_rows,
        variable_q_for_images,
        same_final_images_count,
    )

    false_to_true, true_to_false = compute_correctness_transitions(
        run_data, run_names_in_order, run_meta
    )
    ft_path = os.path.join(args.output_dir, f"{args.output_prefix}_false_to_true.tsv")
    tf_path = os.path.join(args.output_dir, f"{args.output_prefix}_true_to_false.tsv")
    transition_header = [
        "index",
        "correct_answer",
        "question",
        "options",
        "run1_name",
        "run1_answer",
        "run1_correct",
        "run1_final_image_indices",
        "run2_name",
        "run2_answer",
        "run2_correct",
        "run2_final_image_indices",
        "final_image_indices_mismatch",
    ]
    write_tsv(ft_path, transition_header, false_to_true)
    write_tsv(tf_path, transition_header, true_to_false)

    print(f"Processed {len(files)} files.")
    print(f"Saved: {q_path}")
    print(f"Saved: {v_path}")
    print(f"Saved: {r_path}")
    print(f"Saved: {s_path}")
    print(f"Saved: {ft_path}")
    print(f"Saved: {tf_path}")


if __name__ == "__main__":
    main()
