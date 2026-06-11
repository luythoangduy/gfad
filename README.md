# FoundAD <img src="./assets/icon.png" alt="FoundAD logo showing a stylized magnifying glass over abstract shapes, representing anomaly detection in visual data. The logo is set against a neutral background and does not contain any text. The tone is professional and focused." width="30" height="30">

The implementation of the paper **Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors** ([arXiv](http://arxiv.org/abs/2510.01934
), [OpenReview](https://openreview.net/forum?id=YRrlJ8oVEH)).

  <a href="https://ymxlzgy.com/">Guangyao Zhai</a>, <a href="https://karolinezhy.github.io/">Yue Zhou</a>, <a href="">Xinyan Deng</a>, <a href="https://scholar.google.com/citations?user=f5DnPiEAAAAJ&hl=de">Lars Heckler</a>, <a href="https://www.cs.cit.tum.de/camp/members/cv-nassir-navab/nassir-navab/">Nassir Navab</a>, and <a href="https://www.cs.cit.tum.de/camp/members/benjamin-busam/">Benjamin Busam</a>
<br>
  Technical University of Munich <span style="margin: 0 10px;">•</span> MVTec Software GmbH

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Quick Start](#quick-start)
3. [Training and Inference](#train-infer)
   - [Dataset Preparation](#dataset-preparation)
   - [Few-Shot Sampling](#few-shot-sampling)
   - [Model Training](#model-training)
   - [Anomaly Detection / Inference](#anomaly-detection--inference)
4. [Acknowledgement](#acknowledgement)
   

## Environment Setup

All Python dependencies are listed in `requirements.txt`. We recommend Python ≥ 3.10.

```bash
conda create -n foundad python=3.10
conda activate foundad
git clone git@github.com:ymxlzgy/FoundAD.git
cd FoundAD
pip install -r requirements.txt
pip install -e .
```


## Quick Start
Before we start, please make sure you have the rights to use [DINOv3](https://github.com/facebookresearch/dinov3). Download our trained manifold projectors, and put them to `./logs/`. 
|DINOv3-based|1-shot|2-shot|4-shot|
|---------|:---------:|:---------:|:---------:|
|**MVTec AD**|[⬇️ <u>link</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/mvtec_1shot.zip)|[⬇️ <u>link</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/mvtec_2shot.zip)|[⬇️ <u>link</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/mvtec_4shot.zip)|
|**VisA**  |[⬇️ <u>link</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/visa_1shot.zip)|[⬇️ <u>link</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/visa_2shot.zip)|[⬇️ <u>link</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/visa_4shot.zip)|


Run a demo on MVTec-AD 
```bash
python foundad/main.py mode=demo app=test testing.segmentation_vis=True data.dataset=mvtec data.data_name=mvtec_1shot data.test_root=assets/mvtec
```

Or a demo on VisA
```bash
python foundad/main.py mode=demo app=test testing.segmentation_vis=True data.dataset=visa data.data_name=visa_4shot data.test_root=assets/visa
```


## Training and Inference

### Dataset Preparation

| Dataset | Preferred download |
|---------|--------------------|
| **MVTec AD** | Official site: [<u>Here</u>](https://www.mvtec.com/company/research/datasets/mvtec-ad) |
| **VisA** | We use the structured dataset of [<u>RealNet</u>](https://github.com/cnulab/RealNet). |

### Few-Shot Sampling

Create a **few-shot** subset with `sample.py`:

```bash
python foundad/src/sample.py source=/media/ymxlzgy/Data21/xinyan/visa target=/media/ymxlzgy/Data21/xinyan/visa_tmp seed=42 num_samples=2
```
where `source` is the dataset folder, `target` is the folder of few-shot samples, and `num_samples` is the number of samples training models, e.g., 2 for 2-shot learning. `seed` can be adjusted to have multiple rounds of experiment.

### Model Training

```bash
python foundad/main.py mode=train data.batch_size=8 data.dataset=mvtec data.data_name=mvtec_1shot data.data_path=/media/ymxlzgy/Data21/xinyan app=train_dinov3 diy_name=dbug
```
where `data.dataset` is "mvtec" or "visa", `data.data_name` is the folder name of few-shot samples, `data.data_path` is the path where the few-shot folder is at, `app` is "train_dinov3" or other model configs under `configs/app/`, and `diy_name` (optionally) is the post-fix name of the model saving directory. To adjust the layer, please specify `app.meta.n_layer`.

### Anomaly Detection / Inference

After training, run inference:

```bash
python foundad/main.py mode=AD data.dataset=mvtec data.data_name=mvtec_1shot diy_name=dbug data.test_root=/media/ymxlzgy/Data21/xinyan/mvtec app=test app.ckpt_step=1950
```
where `data.test_root` is the dataset folder, and `app` is test_dinov2 or test_dinov3 under `configs/app/`. To adjust sample number K, please specify `testing.K_top_mvtec` and `testing.K_top_visa`.

## Acknowledgement
This repo utilizes [DINOv3](https://github.com/facebookresearch/dinov3), [DINOv2](https://github.com/facebookresearch/dinov2), [DINO](https://github.com/facebookresearch/dino), [SigLIP](https://github.com/google-research/big_vision), [CLIP](https://github.com/openai/CLIP) and [DINOSigLIP](https://github.com/tri-ml/prismatic-vlms). We also thank [I-JEPA](https://github.com/facebookresearch/ijepa) for the inspiration.

## VSCode -> Kaggle Research Workflow

This repository keeps the Kaggle notebook thin and routes training through scripts and configs in the repo. The existing model logic remains under `foundad/`, while the Kaggle runner lives in `notebooks/kaggle_runner/`.

The notebook template is based on an existing working Kaggle flow: clone or reuse the repo, optionally install only missing dependencies, and run a repo script. Dataset attachments should come from Kaggle notebook metadata, so when the notebook is pushed back to Kaggle the linked datasets travel with that metadata.

### Local setup

Create a local `.env` from `.env.example` and fill in Kaggle credentials for each profile you want to use:

```bash
cp .env.example .env
python scripts/select_kaggle_profile.py --profile main
```

The helper scripts read `.env`, resolve the selected profile, and inject `KAGGLE_USERNAME` and `KAGGLE_KEY` into child processes without printing the API key.

Preferred setup for the current Kaggle CLI is an API token per profile:

```env
KAGGLE_PROFILE=main
KAGGLE_MAIN_API_TOKEN=your_main_api_token
```

Legacy `KAGGLE_<PROFILE>_USERNAME` and `KAGGLE_<PROFILE>_KEY` are still supported as a fallback.

### Pull existing Kaggle notebook and metadata

```bash
python scripts/pull_kaggle_kernel.py \
  --profile main \
  --kernel luythoangduy/research-runner
```

Or run the Kaggle CLI directly:

```bash
kaggle kernels pull luythoangduy/research-runner \
  -p notebooks/kaggle_runner \
  -m
```

After pulling, verify `notebooks/kaggle_runner/kernel-metadata.json` and keep the real notebook `id`. The helper normalizes the local code file to `notebook.ipynb` and preserves the existing kernel ID.

If the pulled notebook already contains Kaggle metadata for dataset attachments, keep using that metadata. Do not re-encode dataset links inside notebook cells unless you are intentionally changing the notebook inputs.

Expected metadata shape:

```json
{
  "id": "kaggle_username/kernel_slug",
  "title": "Research Runner",
  "code_file": "notebook.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [
    "dataset_owner/dataset_slug"
  ],
  "competition_sources": [],
  "kernel_sources": []
}
```

Only update the safe fields when needed: `code_file`, `dataset_sources`, `enable_gpu`, `enable_internet`, and `is_private`.

### Push notebook back to Kaggle

```bash
python scripts/push_kaggle_kernel.py --profile main
```

Or directly:

```bash
kaggle kernels push -p notebooks/kaggle_runner
```

If your main Kaggle account is full, switch profiles:

```bash
python scripts/push_kaggle_kernel.py --profile alt1
```

### Run a thin Kaggle experiment

The notebook should only bootstrap the repo, optionally install dependencies if the Kaggle runtime actually needs them, and run an entrypoint such as:

```bash
python scripts/run_experiment.py --config configs/baseline.yaml
```

The sample notebook in `notebooks/kaggle_runner/notebook.ipynb` leaves dependency installation behind a toggle so it works for runners where:

- the Kaggle image already has the required packages
- `pip install -r requirements.txt` is unnecessary
- package installation is restricted or unreliable

The baseline config includes an experiment ID and writes to:

```text
/kaggle/working/outputs/<experiment_id>/
```

The wrapper saves:

```text
config.yaml
metrics.json
git_commit.txt
log.txt
```

Optional artifacts can include `summary.csv`, `result_table.csv`, visualizations, and checkpoints. Checkpoints may stay on Kaggle and should not be pulled back unless you explicitly request them.

### Pull only selected results

```bash
python scripts/pull_kaggle_results.py \
  --profile main \
  --kernel luythoangduy/research-runner \
  --experiment_id exp_001_baseline_seed42
```

This saves files under `experiments/results/exp_001_baseline_seed42/`.

By default the pull script only downloads:

```text
metrics.json
config.yaml
git_commit.txt
summary.csv
result_table.csv
log.txt
```

To include additional small files such as `.png` and `.md`, add:

```bash
python scripts/pull_kaggle_results.py \
  --profile main \
  --kernel luythoangduy/research-runner \
  --experiment_id exp_001_baseline_seed42 \
  --include-optional-small-files
```

To pull checkpoints explicitly:

```bash
python scripts/pull_kaggle_results.py \
  --profile main \
  --kernel luythoangduy/research-runner \
  --experiment_id exp_001_baseline_seed42 \
  --include-checkpoints
```

### Notes

- The dataset should stay on Kaggle and be attached through notebook metadata and `kernel-metadata.json`.
- Do not commit `.env`, `kaggle.json`, datasets, checkpoints, or large outputs.
- If `kaggle` is not on your PATH, install it with `pip install kaggle` before using the helper scripts.
