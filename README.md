# COMP90049 Assignment 2: Early Prediction of Mechanical Failures  
*Group 75 | Elijah Cullinan, Amelia King, Xinyu Xu *  


## Project Overview  
This project investigates **early mechanical failure prediction** using classical machine learning (ML) and neural network baselines, focusing on two core research questions (RQs) to disentangle the impact of environmental vs. internal factors:  
- **RQ1**: To what extent do external environmental factors influence the internal components of mechanical systems such that they cross a point of failure? (Explored via the MetroPT3 dataset)  
- **RQ2**: Can the failure of a mechanical system be predicted based on the manufacturing quality of its internal components? (Explored via the NASA C-MAPSS dataset)  

Key contributions include:  
1. A unified feature engineering framework distinguishing environmental (e.g., MetroPT3 cumulative anomalies) and internal (e.g., NASA thermodynamic ratios) factors.  
2. Comparative evaluation of 5+ models (kNN, Naive Bayes, SVM, Decision Tree, MLP) under consistent validation.  
3. Analysis of prediction horizon reliability for industrial predictive maintenance (PdM) applicability.  


## Datasets  
All datasets are publicly available and preprocessed via custom pipelines (see `src/data_prep/`):  

| Dataset               | Purpose                          | Source Link                                                                 |
|-----------------------|----------------------------------|-----------------------------------------------------------------------------|
| MetroPT3 Air Compressor | RQ1 (Environmental Factors)      | https://archive.ics.uci.edu/dataset/791/metropt+3+dataset                   |
| NASA C-MAPSS Turbofan | RQ2 (Internal/Manufacturing Quality) | https://www.kaggle.com/datasets/behrad3d/nasa-cmaps                        |  


## Core Structure  
The project is organized into these key parts:  

| Directory/File                | What It Does                                                                 |  
|-------------------------------|------------------------------------------------------------------------------|  
| `code/`                       | All Python scripts + Jupyter Notebooks for data processing, modeling, etc.   |  
| `data/`                       | Raw datasets (e.g., NASA turbofan data) + processed data files.              |  
| `docs/`                       | Assignment instructions, literature review notes, and quick references.      |  
| `results/`                    | Model outputs (predictions, metrics) + visualization files.                  |  


## Environment Setup  
To run the code, install the following dependencies (tested versions provided for reproducibility):  

```bash
# Core data processing
pip install pandas==2.2.1 numpy==1.26.4
# Machine learning
pip install scikit-learn==1.4.2 xgboost==2.0.3
# Visualization
pip install matplotlib==3.8.0 seaborn==0.13.2
# Optional: Kaggle dataset download
pip install kagglehub==0.2.5
```  


## Step-by-Step Execution  
Follow these steps to reproduce the project’s key results (focus on NASA/KNN/NB workflow by Jasmine first):  

### 1. Download Raw Data  
- **NASA C-MAPSS**: Run `src/data_prep/nasa_data_loader.py` to auto-download TXT files from GitHub/Kaggle (fallback to local cache if available).  
- **MetroPT3**: Download the dataset from the UCI link above, then place the CSV in `src/data_prep/` (required for `metro_data_loader.py`).  

### 2. Generate Engineered Features  
- **NASA (Jasmine’s Pipeline)**:  
  Run `src/feature_eng/nasa_xinyu_knn_nb.py` (Cells 0–3) to:  
  - Load raw NASA data (FD001–FD004).  
  - Create 4 key feature types: baseline statistics, sensor deviation, rolling slopes, and thermodynamic ratios (e.g., `eff_ratio = pressure_ratio / temp_ratio`).  
  - Save processed features to `features_nasa/` (e.g., `X_train_FD001.csv`, `y_test_FD004.csv`).  

- **MetroPT3**:  
  Run `src/feature_eng/metro_features.py` to generate cumulative anomaly features (e.g., `cum_compressor_eff`).  

### 3. Train & Evaluate Models  
- **NASA KNN/NB (Jasmine’s Work)**:  
  Continue running `src/feature_eng/nasa_xinyu_knn_nb.py` (Cells 4–6) to:  
  - Tune KNN hyperparameters (best: `n_neighbors=6`, `weights="distance"`) via 3-fold CV.  
  - Train Gaussian Naive Bayes (NB) with `var_smoothing=1e-8`.  
  - Evaluate on validation/test sets (e.g., FD001 KNN: MAE=35.91, FD004 NB: RMSE=348.16).  
  - Save predictions to `features_nasa/` (e.g., `pred_FD001.csv`) and metrics to `results/evaluation_summary_nb.csv`.  

- **MetroPT3 Ensemble**:  
  Run `src/model_train/metro_ensemble.py` to train the SVM/kNN/DT ensemble (F1=0.552 for failure prediction).  

### 4. Visualize Results  
- NASA's KNN & NB: The `nasa_xinyu_knn_nb.py` script auto-generates "True vs. Predicted RUL" plots for each FD subset (e.g., FD004 KNN underperformance).  
- MetroPT3: `metro_ensemble.py` outputs confusion matrices and accuracy-vs-instance plots (saved to `results/`).  


## Reproducibility Notes  
- **Random Seeds**: All model training uses `random_state=42` (e.g., `train_test_split`, `GridSearchCV`) to ensure consistent results.  
- **Data Leakage Prevention**:  
  - NASA: Scalers are fit *only* on training data (see `scale_by_train()` in `nasa_xinyu_knn_nb.py`).  
  - MetroPT3: Time-series splits (66-16-18) avoid using future data for training.  
- **Result Validation**: Cross-check model metrics against `results/evaluation_summary_nb.csv` (NB) and Table 1 in the project report (KNN).  


## Contributors  
- **Elijah Cullinan**: MetroPT3 preprocessing, ensemble model training, and Discussion section.  
- **Amelia King**: Literature review, Introduction, MetroPT3 feature design notes，NASA C-MAPSS's RamdonForest, MLP, and idea integration.  
- **Xinyu Xu**: NASA C-MAPSS feature engineering (baseline deviation, thermodynamic ratios), KNN/NB model training/evaluation, and Results section.  


## License  
This project is for educational purposes (COMP90049, University of Melbourne) and uses publicly available datasets. See dataset links for original licensing.
