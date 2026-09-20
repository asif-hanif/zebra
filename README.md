# ZEBRA: Zero-Shot Entropy-Regularized Prompt Learning for Base-to-Novel Generalization in Audio-Language Models (INTERSPEECH'26)

> [**ZEBRA: Zero-Shot Entropy-Regularized Prompt Learning for Base-to-Novel Generalization in Audio-Language Models**](https://arxiv.org/abs/2606.31587)<br><br>
> [Asif Hanif](https://scholar.google.com/citations?hl=en&user=6SO2wqUAAAAJ) and [Mohammad Yaqub](https://scholar.google.com/citations?user=FXJzma8AAAAJ)

<!-- [![page](https://img.shields.io/badge/Project-Page-F9D371)](https://asif-hanif.github.io/trojanwave/) -->
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2606.31587)




<hr />

| ![main figure](/media/zebra.png)|
|:--| 
| **ZEBRA**<p align="justify">ZEBRA approach operates on top of existing prompt learning methods to bridge the base-to-novel generalization gap, preserving zero-shot transferability while benefiting from supervised adaptation through few-shot prompt learning. ZEBRA introduces no additional learnable parameters to existing prompt learning methods and incurs negligible computational overhead.</p> |

</br>


</br>
<hr />
</br>

> **Abstract** <p align="justify"><i>
Audio-Language Models (ALMs) achieve strong zero-shot performance by aligning audio with textual class descriptions. Although prompt learning improves accuracy on base classes through few-shot supervised adaptation, we observe a critical trade-off: it often degrades performance on novel classes, sometimes falling below zero-shot accuracy. This exposes a base-to-novel generalization gap in prompt learning for ALMs. To address this issue, we propose ZEBRA (Zero-shot Entropy-Regularized Prompt Learning for Base-to-Novel Generalization), a plug-and-play framework that fuses zero-shot logits with prompt-learning logits, and employs self-entropy regularization to reduce overfitting to base classes. Experiments across multiple audio classification datasets show that ZEBRA consistently improves novel-class performance while maintaining strong base accuracy, significantly reducing the base-to-novel gap compared to standard prompt learning. 
<br><br>
</i></p>

> <b>TLDR:</b> ZEBRA is a plug-and-play framework for Audio-Language Models that fixes the issue where prompt learning improves known-class accuracy but hurts performance on unseen classes. By fusing zero-shot predictions and applying self-entropy regularization, it significantly boosts accuracy on new categories without requiring any extra learnable parameters.

<br><br>

## Updates :rocket:
- **June 04, 2026** : Accepted in [INTERSPEECH 2026](https://interspeech2026.org/) &nbsp;&nbsp; :confetti_ball: :tada:
- **September 20, 2026** : Released code for ZEBRA


</br>
</br>


## Table of Contents
- [Installation](#installation)
- [Model](#model)
- [Datasets](#datasets)
- [Code Structure](#code-structure)
- [Run Training](#run-training)
- [Run Evaluation](#run-evaluation)
- [Results](#results)
- [Citation](#citation)
- [Contact](#contact)
- [Acknowledgement](#acknowledgement)

</br>
</br>

<a name="installation"/>

## Installation :gear:
1. Create a conda environment
```shell
conda create --name zebra python=3.8
conda activate zebra
```
2. Install PyTorch and other dependencies
```shell
git clone https://github.com/asif-hanif/zebra
cd zebra
pip install -r requirements.txt
```

</br>
<a name="model"/>
    
## Model :white_square_button:

Download the pre-trained PENGI model using the link provided below and place the checkpoint file at path [`pengi/configs`](/pengi/configs) (after clonning the repo). 


| Model | Link | Size |
|:-- |:-- | :-- |
| PENGI | [Download](https://zenodo.org/records/8387083/files/base.pth) | 2.2 GB | 

<br>

PENGI checkpoint can also be downloaded with following command:
```bash
wget https://zenodo.org/records/8387083/files/base.pth
```

</br>

<a name="datasets"/>
    
## Datasets :page_with_curl:

We have performed experiments on 11 audio classification datasets.  Instructions for downloading/processing datasets used by our method have been provided in the [DATASETS.md](DATASETS.md). 

| Dataset | Type | Classes | Size | Link |
|:-- |:-- |:--: |--: |:-- |
| [Beijing-Opera](https://compmusic.upf.edu/bo-perc-dataset) | Instrument Classification | 4 | 69 MB | [Instructions](DATASETS.md#beijing-opera) |
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | Emotion Recognition | 6 | 606 MB | [Instructions](DATASETS.md#crema-d) |
| [ESC50](https://github.com/karolpiczak/ESC-50) | Sound Event Classification | 50 | 881 MB | [Instructions](DATASETS.md#esc50) |
| [ESC50-Actions](https://github.com/karolpiczak/ESC-50) | Sound Event Classification | 10 | 881 MB | [Instructions](DATASETS.md#esc50-actions) |
| [GT-Music-Genre](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) | Music Analysis | 10 | 1.3 GB | [Instructions](DATASETS.md#gt-music-genre) |
| [NS-Instruments](https://magenta.tensorflow.org/datasets/nsynth) | Instrument Classification | 10 | 18.5 GB | [Instructions](DATASETS.md#ns-instruments) |
| [RAVDESS](https://zenodo.org/records/1188976#.YFZuJ0j7SL8) | Emotion Recognition | 8 | 1.1 GB | [Instructions](DATASETS.md#ravdess) |
| [SESA](https://zenodo.org/records/3519845) | Surveillance Sound Classification | 4 | 70 MB | [Instructions](DATASETS.md#sesa) |
| [TUT2017](https://zenodo.org/records/400515) | Acoustic Scene Classification | 15 | 12.3 GB | [Instructions](DATASETS.md#tut2017) |
| [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) | Sound Event Classification | 10 | 6.8 GB | [Instructions](DATASETS.md#urbansound8k) |
| [VocalSound](https://github.com/YuanGongND/vocalsound) | Vocal Sound Classification | 6 | 8.2 GB | [Instructions](DATASETS.md#vocalsound) |

</br>
</br>

All datasets should be placed in a directory named `Audio-Datasets` and the path of this directory should be specified in the variable `DATASET_ROOT` in the shell [`scripts`](/scripts/). The directory structure should be as follows:
```
Audio-Datasets/
    ├── Beijing-Opera/
    ├── CREMA-D/
    ├── ESC50/ 
    ├── ESC50-Actions/
    ├── GT-Music-Genre/
    ├── NS-Instruments/
    ├── RAVDESS/
    ├── SESA/
    ├── TUT2017/
    ├── UrbanSound8K/
    ├── VocalSound/
 ```


</br>

<a name="code-structure"/>

## Code Structure :snowflake:

The repository supports zero-shot evaluation and few-shot prompt learning with **CoOp** and **CoCoOp** on PENGI. [`main.py`](main.py) is the entry point for training and evaluation. The model implementations are in [`methods`](methods), shared training and data utilities are in [`utils`](utils), and the backbone code adapted from [PENGI](https://github.com/microsoft/Pengi) is in [`pengi`](pengi).

**ZEBRA** is enabled with `--use_zebra` for CoOp or CoCoOp. Its zero-shot logit fusion is implemented in [`methods/coop.py`](methods/coop.py) and [`methods/cocoop.py`](methods/cocoop.py), while its self-entropy regularization is implemented in `compute_loss` in [`utils/utils.py`](utils/utils.py).

The main files and folders are shown below; repeated dataset scripts and result files are omitted for brevity.

```text
zebra/
├── main.py                         # Entry point for training and evaluation
├── methods/
│   ├── zeroshot.py                  # Zero-shot PENGI baseline
│   ├── coop.py                      # CoOp prompt learning and optional ZEBRA fusion
│   ├── cocoop.py                    # CoCoOp conditional prompts and optional ZEBRA fusion
│   └── encoders.py                  # Shared audio and text encoders for prompt learning
├── pengi/                           # PENGI backbone and audio preprocessing
├── utils/
│   ├── dataset.py                   # Audio dataset loading and few-shot sampling
│   ├── trainer.py                   # Training and evaluation loops
│   └── utils.py                     # CLI arguments, data loaders, losses, checkpoints, metrics
├── scripts/
│   ├── base-novel-classes/          # Saved base/novel class splits
│   ├── training/**.sh               # Shell scripts to run few-shot training
│   ├── testing/**.sh                # Shell scripts to run evaluation
├── logs/                            # Log files of experiments
```

</br>

<a name="run-experiments"></a>
<a name="run-training"></a>

## Run Training :zap:

Run all commands below from the repository root after completing installation, downloading the PENGI checkpoint, and preparing the datasets as described in [DATASETS.md](DATASETS.md). Experiments were run on an NVIDIA A100-SXM4-40GB GPU.

Before running the scripts, update `DATA_STORAGE_DIR` in the per-dataset scripts under [`scripts/training`](scripts/training) and [`scripts/testing`](scripts/testing). The scripts expect datasets at `$DATA_STORAGE_DIR/datasets/Audio-Datasets` and save/load trained checkpoints under `$DATA_STORAGE_DIR/models/zebra`. Adjust `DATASET_ROOT` and the checkpoint paths if your directory layout differs. These values are assigned inside the shell scripts, so edit the scripts directly.

Training supports `coop` and `cocoop`, with optional ZEBRA enabled by `--use_zebra`. The supplied scripts use 16 examples per base class, 50 epochs, and seed `0`; these settings can be changed in each dataset script. Zero-shot evaluation requires no prompt training.

To train on a single dataset:

```bash
# Syntax: bash scripts/training/<dataset_script>.sh <method> [--use_zebra]

# coop baseline
bash scripts/training/beijing_opera.sh coop

# coop with ZEBRA
bash scripts/training/beijing_opera.sh coop --use_zebra

# cocoop with ZEBRA
bash scripts/training/beijing_opera.sh cocoop --use_zebra
```

Replace `beijing_opera.sh` with another dataset script, such as `esc50.sh` or `vocal_sound.sh`. To train a method on all 11 datasets:

```bash
bash scripts/training/run_all_datasets_coop.sh
bash scripts/training/run_all_datasets_coop.sh --use_zebra
bash scripts/training/run_all_datasets_cocoop.sh
bash scripts/training/run_all_datasets_cocoop.sh --use_zebra
```

Alternatively, run all four configurations with one command:

```bash
bash scripts/training/run_all_datasets_all_methods.sh
```

Checkpoints are named `<DATASET>-SEED<SEED>.pth`, with a `-ZEBRA` suffix for ZEBRA runs, for example `Beijing-Opera-SEED0-ZEBRA.pth`. Training logs and JSON metrics are written to [`logs/training/<METHOD>`](logs/training).

</br>

<a name="run-evaluation"></a>

## Run Evaluation :zap:

Evaluation scripts in [`scripts/testing`](scripts/testing) support `zeroshot`, `coop`, and `cocoop`. Select `base` to evaluate on base classes, `novel` to evaluate on held-out classes, or `all` to evaluate over the combined class set. The default is `novel` when the split argument is omitted; when using `--use_zebra`, explicitly provide the split before the flag.

For CoOp and CoCoOp, first train the matching dataset, method, seed, and ZEBRA configuration. The evaluation script loads its checkpoint from the path configured above. The `zeroshot` baseline uses the pretrained PENGI model without a trained prompt checkpoint.

```bash
# Syntax: bash scripts/testing/<dataset_script>.sh <method> <base|novel|all> [--use_zebra]

# zero-shot baseline on base and novel classes
bash scripts/testing/beijing_opera.sh zeroshot base
bash scripts/testing/beijing_opera.sh zeroshot novel

# coop baseline on base and novel classes
bash scripts/testing/beijing_opera.sh coop base
bash scripts/testing/beijing_opera.sh coop novel

# coop with ZEBRA on base and novel classes
bash scripts/testing/beijing_opera.sh coop base --use_zebra
bash scripts/testing/beijing_opera.sh coop novel --use_zebra

```

To evaluate one configuration across all 11 datasets, use the corresponding wrapper. Replace `novel` with `base` or `all` as needed:

```bash
bash scripts/testing/run_all_datasets_zeroshot.sh novel
bash scripts/testing/run_all_datasets_coop.sh novel
bash scripts/testing/run_all_datasets_coop.sh novel --use_zebra
bash scripts/testing/run_all_datasets_cocoop.sh novel
bash scripts/testing/run_all_datasets_cocoop.sh novel --use_zebra
```

Alternatively, evaluate the zero-shot baseline and both prompt learning methods, with and without ZEBRA, on **base and novel classes separately** across all datasets:

```bash
bash scripts/testing/run_all_datasets_all_methods.sh
```

Evaluation logs and JSON metrics are written to [`logs/testing/<METHOD>`](logs/testing). Filenames include the dataset, seed, evaluation split, and optional `-ZEBRA` suffix, for example `Beijing-Opera-SEED0-NOVEL-ZEBRA.json`. Each JSON records accuracy, F1 score, precision, recall, average loss, and epoch under a seed key such as `seed_0`.

After completing base and novel evaluation for all configurations, aggregate the results and print comparison tables:

```bash
cd logs 
bash results.sh
```

<details>
<summary>Sample Output</summary>
![main figure](/media/zebra_terminal_results.jpg)
</details>

</br>

For multi-fold datasets, the supplied scripts copy the predefined `csv_files/train.csv` and `csv_files/test.csv` into the dataset root, replacing any existing files with those names. These experiments use a single train–test split; they do not run cross-validation.

</br>

<a name="results"/>

## Results :microscope:

<div class="content has-text-justified"><p>


![main figure](/media/zebra_results_banner.jpg)

</br>

![main figure](/media/zebra_results.jpg)

</br>
</br>

<a name="citation"/>

## Citation :star:
If you find our work, this repository, or pretrained models useful, please consider giving a star :star: and citation.
```bibtex
@article{hanif2026zebra,
  title={ZEBRA: Zero-Shot Entropy-Regularized Prompt Learning for Base-to-Novel   Generalization in Audio-Language Models},
  author={Hanif, Asif and Yaqub, Mohammad},
  journal={arXiv preprint arXiv:2606.31587},
  year={2026}
}
```

</br>

<a name="contact"/>

## Contact :mailbox:
Should you have any questions, please create an issue on this repository or contact us at **asif.hanif@mbzuai.ac.ae**

</br>

<a name="acknowledgement"/>

## Acknowledgement :pray:
We used [PENGI](https://github.com/microsoft/Pengi) for model instantiation and borrowed a part of code from CoOp, CoCoOp to implement baselines. We thank the respective authors for releasing the code.

<hr />
