import pandas as pd
from datetime import datetime

df = pd.read_csv('./data/merged_tarkari_dataset.csv', low_memory=False)

# Clean values in minimum, maximum, average fields
cleaning_data = df[['Minimum', 'Maximum', 'Average']]

def clean_price(value):
    if isinstance(value, str):
        value = value.replace('Rs', '').replace(' ', '').replace(',', '')
    try:
        return float(value)
    except:
        return None

for i in cleaning_data:
    df[i] = df[i].apply(clean_price)

# Function that parses the date
def parse_date(date_str):
    for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y'):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

df['Date'] = df['Date'].apply(parse_date)
# Changes the date to required format
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

def clean_unit(unit: str):
    if unit == "Per Dozen" or unit == "Doz":
        return "dozen"
    if unit == "Per Piece" or unit == "Piece":
        return "piece"
    if unit.lower() == "kg":
        return "kg"
    return unit

df["Unit"] = df['Unit'].apply(clean_unit)
df = df.sort_values(by='Date', ascending=True)

# Save complete cleaned data
df.to_csv('./data/cleanDataAll.csv', index=False)

# Define commodities to keep
selected_commodities = [
    'Amla', 'Apple(Fuji)', 'Apple(Jholey)', 'Arum', 'Asparagus',
    'Avocado', 'Bakula', 'Bamboo Shoot', 'Banana', 'Barela'
]

# Filter and save selected data
filtered_df = df[df['Commodity'].isin(selected_commodities)]
filtered_df.to_csv('./data/cleanData.csv', index=False)