# arXiv Research Papers Auto-Tagging

This is a project a part of [Natural Language Proccesing course](https://ods.ai/tracks/nlp-course-spring-2024).

## About

This project proposes a solution to the multilabel classification task. Given a research paper’s title and abstract, the model predicts the arXiv tags associated to the paper (out of 155 tags in total). This work solves the problem by fine-tuning the latest version of DeBERTa. The model reached a peak of ∼64.6% sample-average F1 score.

## Dataset

This project presents a dataset consiting of 536,914 collected research papers stored on arXiv. The collected dataset is published [here](https://www.kaggle.com/datasets/ivanmitriakhin/arxiv-titles-abstracts-and-tags).

## Files

[./data](https://github.com/ivan-mitriakhin/arxiv-auto-tagging/tree/main/data) contains collected data. [to download the data use ```sh ./data/download_data.sh```]

[./docs](https://github.com/ivan-mitriakhin/arxiv-auto-tagging/tree/main/docs) stores the report.

[./notebooks](https://github.com/ivan-mitriakhin/arxiv-auto-tagging/tree/main/notebooks) stores notebooks that contain information about data collection, grouping, etc.

[./src](https://github.com/ivan-mitriakhin/arxiv-auto-tagging/tree/main/src) stores model and utility files as well as the notebook for training the model.

## HF Space

You can utilize the [HF Space](https://huggingface.co/spaces/imitriakhin/deberta-arxiv-tagging) to tag your paper.

## Telegram bot

You can also utilize the [telegram bot](https://t.me/arxiv_tagging_bot) to tag your paper.

An example of inference:

![image](https://github.com/ivan-mitriakhin/arxiv-auto-tagging/blob/main/img/inference.png)