# KTXNN_INT3405E1 — MABe Mouse Behavior Detection

This repository contains the **team solution** for the Kaggle competition  
**MABe Challenge – Social Action Recognition in Mice**.

The project was developed as the **final course project for the Machine Learning course (INT3405E1)** at the **University of Engineering and Technology (UET), VNU Hanoi**.

---

## About the Competition

The **MABe (Multi-Agent Behavior) Challenge** addresses the problem of recognizing and localizing mouse behaviors from **pose trajectories**, rather than raw video frames.  
Each behavior is annotated as a **temporal interval**, which makes the task an event-detection problem instead of standard frame-level classification.

Key challenges of the competition include:
- Multi-animal interactions (up to four mice per video)
- Severe class imbalance, especially for rare social behaviors
- Strong cross-laboratory variability (tracking configurations, frame rates, noise)
- Strict evaluation constraints on valid behaviors and non-overlapping events

The objective is to predict behavior intervals that generalize well across laboratories while respecting the official evaluation protocol.

---

## About This Project

This repository implements a complete end-to-end solution for the MABe Challenge, including:
- Exploratory data analysis and dataset inspection
- Preprocessing of pose-based tracking data
- Feature engineering for single-mouse and pairwise social behaviors
- Model training using gradient boosting methods
- Ensemble inference and event-level post-processing

The implementation emphasizes **robustness, interpretability, and compliance with the official competition rules**, making it suitable for both Kaggle submission and academic reporting.

---

## Repository Structure

KTXNN_INT3405E1/
├── EDA/ # Exploratory data analysis
├── data_raw/ # Utilities for loading raw metadata, tracking, and annotations
├── baselines/ # Single-model baseline experiments
├── ensemble/ # Ensemble training, inference, and post-processing pipeline
├── metric.py # Local evaluation / helper metric utilities
└── .gitignore

---

## Dataset

This project is designed to run in a **Kaggle Notebook environment** using the official dataset:
/kaggle/input/MABe-mouse-behavior-detection

Main data components:
- `train.csv`, `test.csv`
- `train_tracking/`, `test_tracking/` (pose trajectories in Parquet format)
- `train_annotation/` (event-level behavior annotations)

---

## High-Level Method Summary

Our approach follows a pose-to-tabular pipeline:
- Pose trajectories are transformed from long format to frame-level wide format
- Coordinates are normalized using physical scale information (`pix_per_cm_approx`)
- Behaviors are separated into:
  - **Single-mouse behaviors** (self-target actions)
  - **Pairwise social behaviors** (ordered agent–target interactions)
- FPS-aware temporal features are extracted
- One binary classifier is trained per action using:
  - LightGBM
  - XGBoost
  - CatBoost
- Multiple models are combined via probability averaging
- Frame-level predictions are converted into valid event intervals using:
  
  *temporal smoothing → action-wise thresholds → minimum duration → no-overlap constraint*

---

## Academic Context

This work is submitted as the **final project** for:
- **Course:** Machine Learning (INT3405E1)  
- **Institution:** University of Engineering and Technology, VNU Hanoi  

The repository serves both as:
- a reproducible solution for a real-world machine learning challenge, and  
- an academic project demonstrating practical system design under real constraints.

---

## Team Members

- **Nguyen Mai Thanh Thu**  
  University of Engineering and Technology, VNU Hanoi  
  Email: 23021731@vnu.edu.vn

- **Bui Thu Phuong**  
  University of Engineering and Technology, VNU Hanoi  
  Email: 23021667@vnu.edu.vn
