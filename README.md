# Bearing Degradation Analysis & Remaining Useful Life (RUL) Prediction

This repository contains an end-to-end predictive maintenance pipeline for rolling element bearings. Utilizing high-frequency vibration data (such as the NASA Prognostics Center of Excellence dataset), the project extracts critical time-domain features, establishes statistical anomaly thresholds using dimensionality reduction, and implements machine learning models to forecast the Remaining Useful Life (RUL) of structural components.

---

## Project Overview

Accelerated life testing of mechanical bearings generates vast amounts of high-frequency accelerometer data. This project breaks down the analysis into three distinct operational phases:
1. **Signal Processing & Feature Extraction:** Continuous streaming/batch computation of statistical indicators via a custom wrapper class.
2. **Health Indicator (HI) & Thresholding:** Unsupervised health index construction using Principal Component Analysis (PCA) to mark the onset of degradation.
3. **Prognostics:** Regression modeling using Random Forest and Deep Learning architectures to predict exact RUL timelines.

---

## Pipeline & Methodology

### 1. Feature Extraction (`TimeSeriesAnalyzer`)
Raw vibration signals are highly volatile. The custom `TimeSeriesAnalyzer` class ingests raw data files, handles signal cleanup, aggregates the data into structured statistical Key Performance Indicators (KPIs) calculated per file/timestamp:
* **Mean & Standard Deviation (`std`):** Captures general signal shifts and overall energy dispersion.
* **Minimum & Maximum:** Identifies peak transient impacts.
* **Root Mean Square (`rms`):** Tracks the continuous steady-state energy growth of the vibration profile.
* **Kurtosis:** Measures the "spikiness" of the signal—ideal for capturing early-stage subsurface cracks or spalling.

### 2. Anomaly Thresholding via PCA
Instead of manually guessing a threshold on a single noisy metric, this pipeline applies **Principal Component Analysis (PCA)** to the combined **Kurtosis** and **RMS** feature matrix. 
* The first principal component ($PC_1$) acts as a unified **Health Indicator (HI)**.
* A baseline threshold is derived from early-stage (healthy) operational periods, allowing automated detection of when the bearing enters its rapid degradation phase.

### 3. Predictive Modeling for RUL
Once the degradation onset is identified, the target RUL is calculated linearly or exponentially down to failure.
* **Random Forest Regressor:** A robust ensemble ML baseline trained on the extracted time-domain KPIs to map the current degradation state to a remaining time value.
* **LSTM (Long Short-Term Memory) Network _[In Progress]_:** A recurrent deep learning architecture designed to capture sequential dependencies and temporal trends across consecutive time windows, mitigating noise-induced errors in RUL prediction.
