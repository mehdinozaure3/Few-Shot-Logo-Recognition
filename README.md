# Few-Shot Logo Recognition

Project for the course *Machine Learning for Vision and Multimedia*  
Project ID: 2.6 — Few-Shot Logo Recognition

## Goal
This project studies few-shot logo recognition and retrieval using pre-trained visual embeddings and metric learning.  
Given one or a few example images of a logo, the system retrieves all instances of the same logo from a large image gallery, including unseen (novel) logo classes.

## Dataset
We use the **LogoDet-3K** dataset, a large-scale logo detection benchmark with bounding-box annotations.

Dataset homepage: https://github.com/Wangjing1551/LogoDet-3K  
Official download links are provided by the authors (server mirror / Baidu Drive).

We select a manageable subset of logo classes (50–100) with sufficient instances per class.
Each logo instance is cropped from the original image using its bounding box annotation.

## Method
The system follows three phases:
1. **Baseline embeddings** using a pre-trained ResNet (ImageNet), trained with cross-entropy on base classes.
2. **Metric learning** using Prototypical Networks with episodic training for few-shot recognition.
3. **Optional localization** using sliding-window search and embedding similarity.

## Evaluation
We evaluate:
- Classification accuracy on base classes
- Few-shot retrieval on novel classes using Recall@k and mAP
- Episodic accuracy for Prototypical Networks
- Optional localization performance using IoU-based metrics

## Structure
- `data/` : raw and processed datasets
- `src/` : datasets, models, training and evaluation code
- `scripts/` : preprocessing and split generation
- `experiments/` : logs, checkpoints and figures
- `paper/` : final report (LaTeX)

## Reproducibility
All preprocessing, training and evaluation steps are fully scripted and controlled by fixed random seeds.
