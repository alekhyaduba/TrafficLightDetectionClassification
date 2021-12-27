from pandas import DataFrame as df
import pandas as pd
df = pd.DataFrame({
    'name':
    ['orange','banana','lemon','mango','apple'],
    'price':
    [2,3,7,21,11],
    'stock':
    ['Yes','No','Yes','No','Yes']
})
print(df.loc[2]['price'])
print(df.iloc[2]['stock'])