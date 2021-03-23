import pandas as pd

df = pd.read_json('clean_data.json')
df.info()

df['text'][0]