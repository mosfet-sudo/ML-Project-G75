# COMP90049 Assignment 2: Early Prediction of Mechanical Failures  
*Group 75 | Elijah Cullinan, Amelia King, Xinyu Xu*  


## Project Overview  
This project investigates **early mechanical failure prediction** for industrial systems using classical machine learning (ML) and lightweight neural network baselines. It focuses on disentangling the impact of environmental vs. internal factors through two core research questions (RQs), with findings validated against real-world industrial datasets:  
- **RQ1**: To what extent do external environmental factors influence the internal components of mechanical systems such that they cross a point of failure? (Explored via the MetroPT3 air compressor dataset)  
- **RQ2**: Can the failure of a mechanical system be predicted based on the manufacturing quality of its internal components? (Explored via the NASA C-MAPSS turbofan engine dataset)  

Key contributions aligned with the final report:  
1. A **factor-distinct feature engineering framework**: Environmental features (e.g., MetroPT3 cumulative anomaly counts) vs. internal degradation features (e.g., NASA thermodynamic efficiency ratios).  
2. Comparative evaluation of 6 models (kNN, Naive Bayes, SVM, Decision Tree, Random Forest, MLP) under time-series-aware validation to avoid data leakage.  
3. Industrial applicability analysis: Links model performance (e.g., KNN’s RUL prediction error) to practical PdM deployment requirements (e.g., prediction horizon reliability).  


## Datasets  
All datasets are publicly available, with preprocessing pipelines documented in the main notebook (see *Core Structure*). Key details from the final report are：  

| Dataset               | Purpose                          | Key Preprocessing Steps                                                                 | Source Link                                                                 |
|-----------------------|----------------------------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| MetroPT3 Air Compressor | RQ1 (Environmental Factors)      | 10-second → 1-minute downsampling, duplicate removal, normalization (train-set only)     | https://archive.ics.uci.edu/dataset/791/metropt+3+dataset                   |
| NASA C-MAPSS Turbofan | RQ2 (Internal/Manufacturing Quality) | 4-subset segmentation (FD001–FD004), RUL label extraction, sensor outlier filtering | https://www.kaggle.com/datasets/behrad3d/nasa-cmaps                        |  


## Core Structure  
The project follows a modular structure, with the **master notebook** coordinating all workflows (consistent with our actual directory layout):  

| Directory/File                          | What It Does                                                                 |  
|-----------------------------------------|------------------------------------------------------------------------------|  
| `code/90049_A2_G75_code.ipynb`          | Master notebook: Full workflow (data loading → feature engineering → modeling) for both datasets. |  
| `code/nasa_xinyu_knn_nb.ipynb`          | Dedicated module: Xinyu’s NASA feature engineering, KNN/NB training, and visualization. |  
| `data/`                                 | Raw datasets (auto-saved here by loaders) + processed feature CSV files.     |  
| `docs/`                                 | Assignment brief, literature notes (e.g., Li & Ying 2020 on turbofan efficiency), and model tuning logs. |  
| `results/`                              | Model outputs: Prediction CSVs, evaluation metrics, and report-aligned figures (e.g., Figure 3 heatmap). |  


## Environment Setup  
Install dependencies with tested versions (matches report’s reproducibility requirements; adds MLP support):  

```bash
# Core data processing
pip install pandas==2.2.1 numpy==1.26.4
# Machine learning (classical + neural network)
pip install scikit-learn==1.4.2 xgboost==2.0.3 tensorflow==2.15.0 keras==2.15.0
# Visualization (matches report figures)
pip install matplotlib==3.8.0 seaborn==0.13.2
# Optional: Dataset auto-download
pip install kagglehub==0.2.5 requests==2.31.0
```  

*Note*: Use Jupyter Notebook 7.0+ for seamless cell execution (upgrade via `pip install notebook --upgrade` if needed).  


## Step-by-Step Execution  
Follow these steps to reproduce **all report results** (prioritizes Xinyu’s NASA workflow first):  

### 1. Prepare Raw Data  
- **NASA C-MAPSS**: Open `code/nasa_xinyu_knn_nb.ipynb` and run **Cell 0** (auto-downloads FD001–FD004 TXT files to `data/`; uses Kaggle fallback if GitHub is unavailable).  
- **MetroPT3**: Download the dataset from the UCI link above, place `metropt3.csv` in `data/`, then run **Cells 1–2** in `code/90049_A2_G75_code.ipynb` for preprocessing.  

### 2. Generate Engineered Features  
#### NASA (Xinyu’s Module)  
Run **Cells 1–3** in `code/nasa_xinyu_knn_nb.ipynb` to:  
- Load raw sensor data and extract 4 feature types (report Section 3.2):  
  1. Baseline statistics (e.g., `s4_mean` for sensor 4 averages).  
  2. Sensor deviation (e.g., `s15_dev_base` vs. baseline).  
  3. Rolling slopes (e.g., `s7_slope` for degradation trends).  
  4. Thermodynamic ratios (e.g., `eff_ratio = pressure_ratio / temp_ratio`, linked to physical degradation).  
- Save processed features to `data/` (e.g., `X_train_FD001.csv`, `y_test_FD004.csv`).  

#### MetroPT3  
Run **Cells 3–4** in `code/90049_A2_G75_code.ipynb` to generate 8 features (report Section 3.1):  
- Physics-derived features (e.g., `compressor_efficiency`).  
- Cumulative anomaly features (e.g., `cum_overpressure_events` for long-term degradation tracking).  

### 3. Train & Evaluate Models  
#### NASA (Xinyu’s KNN/NB + Amelia’s ML/RF)  
- **KNN/NB**: Continue in `code/nasa_xinyu_knn_nb.ipynb`:  
  - **Cell 4**: Tune KNN via 3-fold CV (best: `n_neighbors=6`, `weights="distance"`; FD001 test MAE=35.91, report Table 1).  
  - **Cell 6**: Train Gaussian NB (`var_smoothing=1e-8`; FD004 test RMSE=348.16, failed due to feature correlation, report Figure 3).  
  - **Cells 5/7**: Save predictions to `results/` (e.g., `pred_FD001.csv`) and metrics to `results/evaluation_summary_nb.csv`.  
- **Random Forest/MLP**: Run **Cells 6–7** in `code/90049_A2_G75_code.ipynb` (Amelia’s contribution; MLP uses 2 hidden layers, report Section 4.2).  

#### MetroPT3 (Elijah’s Ensemble)  
Run **Cells 8–9** in `code/90049_A2_G75_code.ipynb`:  
- Train SVM/kNN/Decision Tree ensemble (weighted voting; test F1=0.552 for failure prediction, report Section 4.1).  
- Save evaluation metrics to `results/metro_ensemble_metrics.csv`.  

### 4. Generate Report-Ready Visualizations  
All scripts auto-produce figures matching the final report:  
- **NASA**: `code/nasa_xinyu_knn_nb.ipynb` outputs "True vs. Predicted RUL" plots (e.g., FD004 KNN underperformance, report Figure 5).  
- **MetroPT3**: `code/90049_A2_G75_code.ipynb` generates confusion matrices (Figure 2) and accuracy-vs-instance plots (Figure 4).  
- **Feature Correlation**: Run **Cell 5** in `code/nasa_xinyu_knn_nb.ipynb` to reproduce Figure 3 (NASA feature heatmap).  


## Reproducibility Notes  
Aligned with report Section 6 (Methodology Rigor):  
- **Random Seeds**: All models use `random_state=42` (e.g., `train_test_split`, `GridSearchCV`) for consistent results.  
- **Data Leakage Prevention**:  
  - NASA: Scalers fit *only* on training data (see `scale_by_train()` in `nasa_xinyu_knn_nb.ipynb`).  
  - MetroPT3: Time-series split (66% train / 16% val / 18% test) to avoid future data leakage.  
- **Result Validation**: Cross-check metrics against:  
  - `results/evaluation_summary_nb.csv` (NASA NB)  
  - Report Table 1 (KNN/RF/MLP)  
  - `results/metro_ensemble_metrics.csv` (ensemble F1-score)  

### Project Limitations (From Report Section 5)  
- NASA: RUL labels rely on single-column extraction (may introduce noise in FD002).  
- MetroPT3: Small sample size limits generalization to other compressor types.  
- Models: MLP performance is constrained by limited training data (no data augmentation applied).  


## Contributors  
- **Elijah Cullinan**: Research question final promotion, MetroPT3 preprocessing, cumulative feature design, SVM/kNN/DT ensemble training, and Discussion section (environmental factor analysis).  
- **Amelia King**: Literature review, Introduction, NASA Random Forest/MLP implementation, and idea integration (RQ alignment promotion).  
- **Xinyu Xu**: Datasets collection and Research question pre-define, NASA C-MAPSS feature engineering (thermodynamic ratios, baseline deviation), KNN/NB model tuning/evaluation, Results section drafting, and visualization (Figure 3/5).  


## License  
This project is for educational purposes (COMP90049, University of Melbourne). Datasets follow original licensing:  
- MetroPT3: UCI Machine Learning Repository Open License.  
- NASA C-MAPSS: Kaggle Public Dataset License.
