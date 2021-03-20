import pandas as pd
import re

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

lookup_tables = [topics_lookup, places_lookup, people_lookup, orgs_lookup, exchanges_lookup]
    

df = pd.read_json('dirty_data.json')

singleelements = ['date', 'topics', 'places', 'people', 'orgs', 'exchanges', 'companies', 'title', 'dateline', 'text', 'author']

for col in singleelements:
    df[col] = df[col].explode()

df['text']=df['text'].replace({r'\s+$': '', r'^\s+': ''}, regex=True).replace(r'\n', '', regex=True)

df.drop('unknown', axis= 1, inplace= True)

#regex/ GREP through topics, people, places, orgs, exchanges
change_cols = df.columns[2:7]
col_index = 0
for col in change_cols:
    current_lookup = lookup_tables[col_index]
    row_index = 0
    for row in df[col]:
        new_row = []
        if row != '':
            for check_word in current_lookup:
                find = re.findall(check_word, row)
                if len(find) >0:
                    new_row.append(find[0])
        df.at[row_index, col] = new_row
        row_index += 1
    col_index += 1

df.to_json('clean_data.json')
