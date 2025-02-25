# [ICLR'25] Jump Your Steps: Optimizing Sampling Schedule of Discrete Diffusion Models

<div align="center">
  <img src="teaser.webp" width="250" alt="Alt text">
</div>

This repository houses the official PyTorch implementation of the paper titled "Jump Your Steps: Optimizing Sampling Schedule of Discrete Diffusion Models", which is presented at ICLR 2025. 

- [Arxiv](https://arxiv.org/abs/2410.07761)
- [OpenReview](https://openreview.net/forum?id=pD6TiCpyDR)

Contacts:

- Yonghyun Park: enkeejunior1@snu.ac.kr 

> **[Notice]** This repository contains all necessary code to reproduce the experimental results for countdown, CIFAR-10, and text experiments. However, this repository has been archived. For active development and support, please refer to the maintained repository at [JYS](https://github.com/enkeejunior1/jump-your-steps).

---

# TL;DR

Jump Your Steps (JYS) is a method for optimizing the sampling schedule of discrete diffusion models. This enables improved sample quality **without increasing computational cost during inference**. Our method demonstrated performance improvements regardless of the data type (image, piano note, text) or transition kernel (gaussian, uniform, absorb).

---

# Prerequisites

1. Setup

```bash
conda create -n jys python=3.10
conda activate jys
pip install -r requirements.txt
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install flash-attn --no-build-isolation # Flash Attention (for SEDD language model)
```

Since torch-fidelity does not accept saved tensor, you have to install it manually.
If you do not want to use automatic evaluation, you can skip this step.

```bash
git clone https://github.com/enkeejunior1/torch-fidelity.git # Torch-Fidelity (for image generation evaluation)
cd torch-fidelity
pip install -e .
cd ..
```

2. Download the pretrained models and datasets.

For CIFAR10 and Piano roll experiment, you have to download the cifar (model) and piano (model, dataset) from the [tauLDR repository](https://www.dropbox.com/scl/fo/zmwsav82kgqtc0tzgpj3l/h?dl=0&rlkey=k6d2bp73k4ifavcg9ldjhgu0s) manually.

Place the downloaded files in their respective directories:

```
jump-your-steps
│
├── weights
│   ├── cifar-ctmc
│   │   ├── ckpt.pt
│   │   └── config.yaml
│   ├── piano-ctmc
│   │   ├── ckpt.pt
│   │   └── config.yaml
│   └── count-sedd
│       └── ckpt.pt
│   
├── datasets
│   └── piano
│       ├── test.npy
│       └── train.npy
│   
├── train
│   ├── unlearn-sd.py
│   └── unlearn-sd3.py
│   
└── scripts
    ├── sd-violence.sh
    ├── sd-nudity.sh
    ├── sd-baseline.sh
    └── sd3-nudity.sh
```

3. (Optional) Train the models for CountDown dataset.

To train models for the CountDown dataset, run:

```bash
bash scripts/train.sh
```

It takes around less than 5 minutes to train the models.

---

# Experiment

Now, let's optimize the sampling schedule using the Jump Your Steps algorithm.

```bash
bash scripts/$dataset_name.sh
```

For `$dataset_name`, enter the desired dataset name. You can choose one from: count, cifar, or text.

---

# Visualization

For the visualization of the results, please refer to `figures.ipynb`.

---

# References

This project is heavily based on the following codes:

- [CTMC](https://github.com/andrew-cr/tauLDR/tree/main)  
- [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion)

---

# Bibtex

```
@inproceedings{
park2025jump,
title={Jump Your Steps: Optimizing Sampling Schedule of Discrete Diffusion Models},
author={Yong-Hyun Park and Chieh-Hsin Lai and Satoshi Hayakawa and Yuhta Takida and Yuki Mitsufuji},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=pD6TiCpyDR}
}
```
