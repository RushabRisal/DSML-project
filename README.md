## Guide to get started

1. Start by creating a virtual env in python
2. Install all required packages (from requirements.txt)
3. Run /scripts/cleanData.py -> You'll get a new "cleanedData.csv" in /data
4. Create folder "price_trends_box" and "price_trends_line" inside "visualizationFig" folder if you don't have one.
5. Run `scripts/analysis/price_trend.py`
6. Run `scripts/analysis/season_analysis.py`
7. Run `scripts/training/time_series_model.py`
8. Run `scripts/training/commodity_clustering.py`
9. Run `scripts/training/train_classification_model.py`
10. Run `streamlit run app.py`

## Commodities to check:
selected_commodities = [
    'Amla', 'Apple(Fuji)', 'Apple(Jholey)', 'Arum', 'Asparagus',
    'Avocado', 'Bakula', 'Bamboo Shoot', 'Banana', 'Barela'
]