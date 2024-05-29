import pandas as pd

df = pd.read_csv('./~1500.csv')

# for a in df['picture']:
#     print(a)
df['longitude'] = df['longitude'].apply(lambda x: x.split('°')[0] if 'E' in x.split('°')[1] else -1*float(x.split('°')[0]))
df['latitude'] = df['latitude'].apply(lambda x: x.split('°')[0])
print(df)

df.to_csv('./~1500.csv', index=False)