# Bot Detection – Social Media Competition

This repository contains our solution for the Bot Detection Competition. The objective of this task is to identify automated accounts (bots) from social media datasets containing user profiles and their associated posts.

## Overview

Our approach models bot detection as a supervised binary classification problem at the user level. Each user is represented through a combination of behavioral, textual, and semantic features extracted from their posts and metadata.

The final system relies on an ensemble architecture combining three complementary signals: tabular features derived from user behavior, textual representations based on TF-IDF, and semantic representations obtained using sentence embeddings. These signals are aggregated through a stacking model trained to optimize a competition-specific scoring function.

## Data Processing

The datasets consist of JSON files containing users and their associated posts. For each user, we aggregate all available posts and compute a set of descriptive features.

The tabular features capture behavioral patterns such as tweet frequency, temporal regularity, burstiness, lexical diversity, duplication patterns, and usage of hashtags, mentions, and URLs. Profile-related features such as username characteristics, description length, and location information are also included.

For textual modeling, all tweets of a user are concatenated into a single document. This document is used to build TF-IDF representations capturing lexical patterns.

In addition, semantic embeddings are computed using a multilingual SentenceTransformer model. The embedding of a user is obtained by averaging the embeddings of their individual posts.

## Model Architecture

The final model is a stacking ensemble composed of three base models and a meta-learner.

The first base model is a LightGBM classifier trained on tabular features. The second model is a Logistic Regression classifier trained on TF-IDF features. The third model is a Logistic Regression classifier trained on sentence embeddings.

The outputs of these three models are probabilities representing the likelihood of a user being a bot. These probabilities are then used as input to a meta-model (Logistic Regression), which produces the final prediction.

## Threshold Selection

Predictions are converted into binary decisions using a threshold optimized on validation data. Unlike standard classification tasks, the competition uses a custom scoring function defined as:

Score = 2 × TP − 2 × FN − 6 × FP

Because false positives are heavily penalized, the threshold is selected to favor high precision and reduce the risk of incorrectly flagging human accounts.

The optimal threshold is determined using a leave-one-dataset-out validation strategy, ensuring robustness across different datasets.

## Language Handling

The solution supports both English and French datasets. A global TF-IDF model is trained on all data, and additional language-specific TF-IDF models are trained when sufficient data is available.

At inference time, the system automatically selects the appropriate TF-IDF model based on the dataset language. If no specific model is available, the global model is used as a fallback.

## Evaluation Strategy

Model selection and validation are performed using a leave-one-dataset-out approach. In this setup, one dataset is held out for validation while the model is trained on the remaining datasets. This process is repeated across all datasets to ensure generalization.

The final model and threshold are trained using all available datasets.

## Output Format

The system produces a text file containing the user IDs of accounts classified as bots. Each user ID appears on a separate line. The format follows the competition requirements and is identical to the provided dataset.bots.txt files.

## Reproducibility

The code is fully deterministic given a fixed random seed. All preprocessing steps, feature engineering, model training, and inference pipelines are included in the repository.

## Notes

The model is designed to be conservative, prioritizing precision over recall due to the asymmetric competition scoring. This reduces the number of false positives, which are significantly penalized.

The approach is robust across datasets and languages and does not rely on dataset-specific tuning.