import pandas as pd

def generateLookup(path):
    output = []
    file = open(path, 'r')
    lines = file.readlines()
    
    for line in lines:
        line = line.replace('\n', '')
        output.append(line)
    
    file.close()
    return output

topics_lookup = generateLookup('raw_data/all-topics-strings.lc.txt')
places_lookup = generateLookup('raw_data/all-places-strings.lc.txt')
people_lookup = generateLookup('raw_data/all-people-strings.lc.txt')
orgs_lookup = generateLookup('raw_data/all-orgs-strings.lc.txt')
exchanges_lookup = generateLookup('raw_data/all-exchanges-strings.lc.txt')
    

df = pd.read_json('dirty_data.json')

singleelements = ['date', 'topics', 'places', 'people', 'orgs', 'exchanges', 'companies', 'title', 'dateline', 'text', 'author']

for col in singleelements:
    df[col] = df[col].explode()

df.loc[df['unknown'].isna(), 'unknown'] = ['']
df.loc[df['unknown'].notna(), ['unknown1','unknown2']] = pd.DataFrame(df.loc[df['unknown'].notna(),'unknown'].tolist(), index= df.index)

df['text']=df['text'].replace({r'\s+$': '', r'^\s+': ''}, regex=True).replace(r'\n', '', regex=True)

df.drop('unknown', axis= 1, inplace= True)

#regex/ GREP through topics, people, places, orgs, exchanges
