import pandas as pd
import matplotlib.pyplot as plt #type:ignore
import seaborn as sns
from scipy.stats import zscore
#import the data and analyzing the whether it is completedly cleaned or not

data=pd.read_csv('./data/cleanData.csv')
# print(data.head())
print(data[['Minimum','Maximum','Average']].describe())
inconsitent_row  = data[(data['Minimum'] > data['Average']) | (data['Maximum'] < data['Average'])]
print(f'The inconsistent row is: {inconsitent_row.count()}')

#checking the outliers
visual_data= data[['Minimum','Maximum','Average']]
sns.boxenplot(data = visual_data)
plt.savefig('./visualizationFig/boxplot.png')
#finding out the outliers 
data['zscore_min'] = zscore(data['Minimum'])
data['zscore_max'] = zscore(data['Maximum'])
data['zscore_avg'] = zscore(data['Average'])

outliers = data[(data['zscore_min'].abs() > 3) | (data['zscore_max'].abs() > 3) | (data['zscore_avg'].abs() > 3)]
print(f'The outliears are: {outliers.count()}')

#checking the duplicate values
duplicate = data[data.duplicated()]
print(f'the duplicate values are: {duplicate}')

#checking the distribution
# data[['Minimum','Maximum','Average']].hist(bins=40, figsize=(10,5))
# plt.savefig('./visualizationFig/histogram.png')

before_data=data[['Date','Average']]
plt.scatter(before_data['Date'],before_data['Average'])
plt.savefig('./visualizationFig/beforePlot.png')