import pandas as pd

data = pd.read_csv('/home/heemin/GFL/data_r50.csv')
data.groupby('malicious').mean()