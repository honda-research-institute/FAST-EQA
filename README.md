# FAST-EQA

**FAST-EQA: Efficient Embodied Question Answering with Global and Local Region Relevancy**

Haochen Zhang<sup>1</sup>,
Nirav Savaliya<sup>2</sup>,
Faizan Siddiqui<sup>2</sup>,
Enna Sachdeva<sup>2</sup>

<sup>1</sup>Carnegie Mellon University &nbsp;&nbsp; <sup>2</sup>Honda Research Institute USA

*IEEE/CVF Winter Conference on Applications of Computer Vision (**WACV**), 2026*

[**Paper**](https://openaccess.thecvf.com/content/WACV2026/papers/Zhang_FAST-EQA_Efficient_Embodied_Question_Answering_with_Global_and_Local_Region_WACV_2026_paper.pdf)
&nbsp;|&nbsp; [**Project Page**](https://astronirav.github.io/fasteqa/)

---

FAST-EQA is an Embodied Question Answering framework in which an agent navigates
an unseen 3D environment to gather the visual evidence needed to answer a
natural-language question. It pairs **question-conditioned global exploration**
with **target-aware local search**, maintains a bounded visual memory, and uses
chain-of-thought reasoning to answer faster and more accurately while scaling to
long-horizon tasks.

The agent runs in [Habitat-Sim](https://github.com/facebookresearch/habitat-sim)
over HM3D scenes. It maintains a TSDF-based occupancy and semantic-value map,
proposes frontier and door candidates for exploration, scores them with CLIP /
SigLIP relevancy against the question, and queries a VLM at each step to decide
both whether to keep exploring and what the final answer is.

## Installation

The environment is pinned for Python 3.9 / PyTorch 2.2.1 / CUDA.

```bash
conda env create -f environment.yml
conda activate explore-eqa
pip install -e .
```

`habitat-sim` must be installed separately following the
[official instructions](https://github.com/facebookresearch/habitat-sim#installation);
it is not installable from the environment file on all platforms.

The `CLIP/` and `prismatic-vlms/` directories are vendored dependencies and are
used directly from the repository — no separate install is required.

## Credentials

No API keys are stored in this repository. Both are read from the environment:

```bash
export HF_TOKEN=<your huggingface token>      # gated VLM weights (Prismatic / Llama-2)
export OPENAI_API_KEY=<your openai key>       # only if using a GPT model as the VLM
```

`HF_TOKEN` is consumed by the configs via `${oc.env:HF_TOKEN,""}`.

## Data

### Scenes

Download the [HM3D](https://aihabitat.org/datasets/hm3d-semantics/) scene
datasets and point each config's `scene_data_path` at your local copy. Every
config ships with the placeholder `/path/to/scene_datasets/...`, which **must**
be edited before running.

### Questions

Question and initial-pose files are included under `data/`:

| File | Benchmark |
| --- | --- |
| `aeqa_questions-184*.json`, `open-eqa-v0-hm3d*.json` | A-EQA / OpenEQA |
| `questions.csv` | HM-EQA |
| `MT-HM3D-filtered-new*.csv` | MT-HM3D |
| `express-bench.json` | EXPRESS-Bench |
| `scene_init_poses*.csv` | Agent initial poses |

## Running

Each experiment is driven by a single OmegaConf YAML file:

```bash
python run_vlm_explore.py -cf cfg/vlm_exp_aeqa184.yaml
```

| Config | Benchmark |
| --- | --- |
| `cfg/vlm_exp_aeqa184.yaml` | A-EQA (184-question subset) |
| `cfg/vlm_exp_openeqa.yaml` | OpenEQA (HM3D split) |
| `cfg/vlm_exp_hmeqa.yaml` | HM-EQA |
| `cfg/vlm_exp_mthm3d.yaml` | MT-HM3D |
| `cfg/vlm_exp_express.yaml` | EXPRESS-Bench |

Useful config fields:

- `scene_data_path` — path to your HM3D scenes (**edit this first**)
- `vlm.device` — CUDA device for the VLM, e.g. `cuda:0`
- `vlm.model_id` / `vlm.model_name` — VLM backbone (e.g. `prism-dinosiglip+7b`)
- `start_idx` / `end_idx` — question range, for sharding runs across GPUs
- `output_parent_dir` / `exp_name` — results land in `<output_parent_dir>/<exp_name>/`
- `save_obs` — dump per-step observation images (slow, useful for debugging)

Logs are written to `<output_parent_dir>/<exp_name>/log<start_idx>_<end_idx>.log`.

## Evaluation

Metrics are parsed back out of the run logs:

```bash
# Multiple-choice benchmarks (HM-EQA, MT-HM3D)
python eval_results_on_log.py --dataset HM-EQA --data_path data/questions.csv

# Open-vocabulary benchmarks (A-EQA, OpenEQA, EXPRESS)
python eval_results_on_log.py --dataset A-EQA \
    --data_path data/aeqa_questions-184-gtdist.json --is_open_answer
```

## Repository layout

```
run_vlm_explore.py     Main exploration + QA loop
explore.py             Exploration policy helpers
relevancy.py           CLIP/SigLIP question-image relevancy scoring
am_radio.py            AM-RADIO feature backbone wrapper
featurizer.py          Feature extractor interface
dataset.py             Benchmark loaders
eval_results_on_log.py Metric computation from run logs
src/tsdf.py            TSDF volume + frontier/value planner
src/habitat.py         Habitat-Sim setup and coordinate transforms
src/geom.py            Geometry utilities
src/vlm.py             VLM wrappers (Prismatic, GPT)
cfg/                   Per-benchmark experiment configs
data/                  Question sets and initial poses
script/                One-off dataset-prep and smoke-test scripts
CLIP/                  Vendored OpenAI CLIP (MIT)
prismatic-vlms/        Vendored Prismatic VLMs (MIT)
```

Note: files in `script/` are exploratory dataset-preparation utilities with
hardcoded placeholder paths (`"?"`); they are provided for reference and need
editing before use.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zhang2026fasteqa,
  title     = {FAST-EQA: Efficient Embodied Question Answering with Global and Local Region Relevancy},
  author    = {Zhang, Haochen and Savaliya, Nirav and Siddiqui, Faizan and Sachdeva, Enna},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2026}
}
```

## License

BSD 2-Clause — see [LICENSE](LICENSE). Vendored third-party code retains its own
license; see the notices in `LICENSE`, `CLIP/LICENSE`, and
`prismatic-vlms/LICENSE`.

## Acknowledgements

Built on explore-eqa (Allen Ren et al., Princeton University), [tsdf-fusion-python](https://github.com/andyzeng/tsdf-fusion-python)
(Andy Zeng), [CLIP](https://github.com/openai/CLIP) (OpenAI), and
[Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms) (TRI / Stanford).
