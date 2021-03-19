import pandas as pd

df = pd.read_json('dirty_data.json')

singleelements = ['date', 'topics', 'places', 'people', 'orgs', 'exchanges', 'companies', 'title', 'dateline', 'text', 'author']

for col in singleelements:
    df[col] = df[col].explode()

df.loc[df['unknown'].isna(), 'unknown'] = ['']
df.loc[df['unknown'].notna(), ['unknown1','unknown2']] = pd.DataFrame(df.loc[df['unknown'].notna(),'unknown'].tolist(), index= df.index)

df['text']=df['text'].replace({r'\s+$': '', r'^\s+': ''}, regex=True).replace(r'\n', '', regex=True)

df.drop('unknown', axis= 1, inplace= True)

#regex/ GREP through topics, people, places, orgs, exchanges
