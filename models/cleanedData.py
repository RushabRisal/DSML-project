import pandas as pd

df=pd.read_csv('./data/tarkari.csv')

#first we will clean the value as in the field [minimum, maximum, average] is not uniform.
cleaning_data= df[['Minimum','Maximum','Average']]

#defining the function that removes all the str, space, and comma and converts to float or int 
def clean_data(value):
    if isinstance(value, str): #here the isinstance checks if the value is string or not
        value = value.replace('Rs', '').replace(' ', '').replace(',', '')
    try:
        return float(value)
    except:
        return None

for i in cleaning_data:
    df[i] = df[i].apply(clean_data)


df.to_csv('./data/cleanedData.csv', index=False)
