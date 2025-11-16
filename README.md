# DSML-project

A small data science / machine learning project that prepares, analyzes and models agricultural commodity price data and exposes an interactive Streamlit app for exploration and forecasts.

## Table of contents
- Project overview
- Repository structure
- Quick start
- Detailed setup & pipeline
- Scripts (what they do)
- Files & outputs produced
- Running the Streamlit app
- Common issues & troubleshooting
- Notes & contact

## Project overview
This repository contains code to:
- Clean and preprocess commodity pricing data
- Produce exploratory visualizations and seasonal analyses
- Train time-series forecasting models and classification/clustering models
- Provide an interactive Streamlit app (app.py) to explore data, visualizations and model outputs

## Repository structure
- .streamlit/                    - Streamlit configuration (UI / theming)
- app.py                         - Streamlit application (entry point)
- data/                          - Input and output data (e.g., cleanedData.csv)
- models/                        - Saved trained models and artifacts
- scripts/                       - Data cleaning, analysis, visualization and training scripts
  - scripts/analysis/             - Exploratory analysis scripts (price trends, season analysis)
  - scripts/cleaning/             - Visualization utilities used during cleaning/EDA
  - scripts/training/             - Training scripts for forecasting, clustering, classification
- visualizationFig/              - Generated figures (create subfolders if missing)
  - visualizationFig/daily_price_trend/ (kept with README so folder persists)
- requirements.txt               - Python dependencies
- README.md                      - This file

## Quick start
1. Clone the repo:
   git clone https://github.com/RushabRisal/DSML-project.git
2. Create and activate a virtual environment:
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate      # Windows
3. Install dependencies:
   pip install -r requirements.txt
4. Follow the pipeline (see Detailed setup & pipeline) or run the Streamlit app:
   streamlit run app.py

## Detailed setup & pipeline
1. Prepare environment and install dependencies (see Quick start).
2. Run the cleaning script to produce cleaned data:
   python scripts/cleanData.py
   - This should create `cleanedData.csv` inside the `data/` folder.
3. Ensure visualization output folders exist (if not, create them):
   - visualizationFig/price_trends_box
   - visualizationFig/price_trends_line
4. Run exploratory & visualization scripts:
   - python scripts/analysis/price_trend.py
   - python scripts/cleaning/visuals.py
   - python scripts/analysis/season_analysis.py
   These generate figures under `visualizationFig/`.
5. Train models:
   - python scripts/training/time_series_model.py
   - python scripts/training/commodity_clustering.py
   - python scripts/training/train_classification_model.py
   Trained model files / artifacts will be saved to `models/`.
6. Launch the Streamlit app:
   streamlit run app.py

## Scripts (high-level descriptions)
- scripts/cleanData.py
  - Cleans & standardizes the raw dataset and outputs `data/cleanedData.csv`.
- scripts/analysis/price_trend.py
  - Plots price trends for selected commodities (daily/aggregate).
- scripts/cleaning/visuals.py
  - Additional visualization helpers used in EDA.
- scripts/analysis/season_analysis.py
  - Performs seasonal analysis across months/years to identify seasonal patterns.
- scripts/training/time_series_model.py
  - Trains time-series forecasting models for commodity prices.
- scripts/training/commodity_clustering.py
  - Performs clustering on commodities (e.g., KMeans) to find groups with similar behavior.
- scripts/training/train_classification_model.py
  - Trains classification models (task depends on repo’s labeling logic).

## Example commodity list
The app and some analysis scripts use a selected list of commodities, for example:
selected_commodities = [
    'Amla', 'Apple(Fuji)', 'Apple(Jholey)', 'Arum', 'Asparagus',
    'Avocado', 'Bakula', 'Bamboo Shoot', 'Banana', 'Barela'
]

## Files & outputs produced
- data/cleanedData.csv         — cleaned/processed dataset used by analyses & models
- visualizationFig/*           — PNGs / images produced by analysis scripts
- models/*                     — serialized model artifacts and model metadata

## Running the Streamlit app
1. Ensure `data/cleanedData.csv` exists (run cleaning script if needed).
2. From repository root:
   streamlit run app.py
3. The app will open in the browser (http://localhost:8501 by default). Use the UI to explore commodities, visualizations and model outputs.

## Common issues & troubleshooting
- Missing folders: create required folders (e.g., visualizationFig/price_trends_box and price_trends_line) if scripts expect them.
- Dependencies: if installation fails, ensure pip, Python version (recommended 3.8+) and virtual environment are set up correctly.
- Data not found: confirm cleaned CSV exists at data/cleanedData.csv or adjust script paths.
- Long-running training: training scripts may take time depending on dataset size and model complexity. Consider using a smaller subset while testing.

## Notes
- This README documents how to get started and the pipeline with the scripts currently present in the repository. Exact hyperparameters, model types and plot formats are defined within the individual scripts.
- There is no LICENSE file in this repo. Check the repository owner for usage/redistribution permissions.

