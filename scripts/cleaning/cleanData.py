import pandas as pd
from datetime import datetime

df=pd.read_csv('./data/merged_tarkari_dataset.csv', low_memory=False)

#first we will clean the value as in the field [minimum, maximum, average] is not uniform.
cleaning_data= df[['Minimum','Maximum','Average']]

#defining the function that removes all the str, space, and comma and converts to float or int 
def clean_price(value):
    if isinstance(value, str): #here the isinstance checks if the value is string or not
        value = value.replace('Rs', '').replace(' ', '').replace(',', '')
    try:
        return float(value)
    except:
        return None

for i in cleaning_data:
    df[i] = df[i].apply(clean_price)


#function that parses the date
def parse_date(date_str):
    for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y'):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

df['Date'] = df['Date'].apply(parse_date)
#changes the date to requried format
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')


def clean_unit(unit: str):
    if unit == "Per Dozen" or unit == "Doz":
        return "dozen"
    if unit == "Per Piece" or unit == "Piece":
        return "piece"
    if unit.lower() == "kg" :
        return "kg"
    return unit

df["Unit"] = df['Unit'].apply(clean_unit)

# Drop duplicates if present
df = df.drop_duplicates()

df.to_csv('./data/cleanData.csv', index=False)
