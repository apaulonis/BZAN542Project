

import pandas as pd
#import plotly.express as px
import random


df = pd.read_json('clean_data.json')
#df.info()

"""
This part generates a dictionary used to index the topics we have into the 6 broad categories listed in
"cat-descriptions_120396.txt"

"""

"""
First pass at categories, based on docs


s_money_forex = {"MONEY-FX", "SHIP", "INTEREST"}
s_economic_indicator = {"BOP", "TRADE", "CPI", "WPI", "JOBS", "IPI", "CPU", "GNP","MONEY-SUPPLY", "RESERVES", "LEI","HOUSING","INCOME","INVENTORIES","INSTAL-DEBT"," RETAIL"}
s_currency = {"DLR","AUSTDLR","HK","SINGDLR","NZDLR","CAN","STG","DMK","YEN","SFR","FFR","BFR","DFL","LIT","DKR","NKR","SKR","MEXPESO","CRUZADO","AUSTRAL","SAUDRIYAL","RAND","RUPIAH","RINGGIT","ESCUDO","PESETA","DRACHMA"}
s_corporate = {"ACQ", "EARN"}
s_commodity = {"ALUM","BARLEY","CARCASS","CASTOR-MEAL","CASTOR-OIL","CASTORSEED","CITRUSPULP","COCOA","COCONUT-OIL","COCONUT","COFFEE","COPPER","COPRA-CAKE","CORN-OIL","CORN","CORNGLUTENFEED","COTTON ","COTTON-MEAL","COTTON-OIL","COTTONSEED","F-CATTLE","FISHMEAL","FLAXSEED","GOLD","GRAIN","GROUNDNUT","GROUNDNUT-MEAL","GROUNDNUT-OIL","IRON-STEEL","LEAD","LIN-MEAL","LIN-OIL","LINSEED","LIVESTOCK","L-CATTLE","HOG","LUMBER","LUPIN","MEAL-FEED","NICKEL","OAT","OILSEED","ORANGE","PALLADIUM","PALM-MEAL","PALM-OIL","PALMKERNEL","PLATINUM","PLYWOOD","PORK-BELLY","POTATO","RAPE-MEAL","RAPE-OIL","RAPESEED","RED-BEAN","RICE","RUBBER","RYE","SILK","SILVER","SORGHUM","SOY-MEAL","SOY-OIL","SOYBEAN","STRATEGIC-METAL","SUGAR","SUN-MEAL","SUN-OIL","SUNSEED","TAPIOCA","TEA","TIN","TUNG-OIL","TUNG","VEG-OIL","WHEAT","WOOL","ZINC"}
s_energy = {"CRUDE","HEAT","FUEL","GAS","NAT-GAS","PET-CHEM","PROPANE","JET","NAPHTHA"}

s_money_forex = {x.lower() for x in s_money_forex}
s_economic_indicator = {x.lower() for x in s_economic_indicator}
s_currency = {x.lower() for x in s_currency}
s_corporate = {x.lower() for x in s_corporate}
s_commodity = {x.lower() for x in s_commodity}
s_energy = {x.lower() for x in s_energy}

subtopics = {'money': s_money_forex, 
            'econ': s_economic_indicator, 
            'currency': s_currency, 
            'corp': s_corporate,
            'commodity': s_commodity,
            'energy': s_energy}

"""

"""
Combining categories/ Splitting Corporate
"""
s_money_forex = {"MONEY-FX", "SHIP", "INTEREST", "DLR","AUSTDLR","HK","SINGDLR","NZDLR","CAN","STG","DMK","YEN","SFR","FFR","BFR","DFL","LIT","DKR","NKR","SKR","MEXPESO","CRUZADO","AUSTRAL","SAUDRIYAL","RAND","RUPIAH","RINGGIT","ESCUDO","PESETA","DRACHMA"}
s_economic_indicator = {"BOP", "TRADE", "CPI", "WPI", "JOBS", "IPI", "CPU", "GNP","MONEY-SUPPLY", "RESERVES", "LEI","HOUSING","INCOME","INVENTORIES","INSTAL-DEBT"," RETAIL"}
s_acquisitions = {"ACQ"}
s_earnings = {"EARN"}
s_commodity = {"ALUM","BARLEY","CARCASS","CASTOR-MEAL","CASTOR-OIL","CASTORSEED","CITRUSPULP","COCOA","COCONUT-OIL","COCONUT","COFFEE","COPPER","COPRA-CAKE","CORN-OIL","CORN","CORNGLUTENFEED","COTTON ","COTTON-MEAL","COTTON-OIL","COTTONSEED","F-CATTLE","FISHMEAL","FLAXSEED","GOLD","GRAIN","GROUNDNUT","GROUNDNUT-MEAL","GROUNDNUT-OIL","IRON-STEEL","LEAD","LIN-MEAL","LIN-OIL","LINSEED","LIVESTOCK","L-CATTLE","HOG","LUMBER","LUPIN","MEAL-FEED","NICKEL","OAT","OILSEED","ORANGE","PALLADIUM","PALM-MEAL","PALM-OIL","PALMKERNEL","PLATINUM","PLYWOOD","PORK-BELLY","POTATO","RAPE-MEAL","RAPE-OIL","RAPESEED","RED-BEAN","RICE","RUBBER","RYE","SILK","SILVER","SORGHUM","SOY-MEAL","SOY-OIL","SOYBEAN","STRATEGIC-METAL","SUGAR","SUN-MEAL","SUN-OIL","SUNSEED","TAPIOCA","TEA","TIN","TUNG-OIL","TUNG","VEG-OIL","WHEAT","WOOL","ZINC"}
s_energy = {"CRUDE","HEAT","FUEL","GAS","NAT-GAS","PET-CHEM","PROPANE","JET","NAPHTHA"}

s_money_forex = {x.lower() for x in s_money_forex}
s_economic_indicator = {x.lower() for x in s_economic_indicator}
s_acquisitions = {x.lower() for x in s_acquisitions}
s_earnings = {x.lower() for x in s_earnings}
s_commodity = {x.lower() for x in s_commodity}
s_energy = {x.lower() for x in s_energy}

subtopics = {'money': s_money_forex, 
            'econ': s_economic_indicator, 
            'acquisitions': s_acquisitions,
            'earnings' : s_earnings,
            'commodity': s_commodity,
            'energy': s_energy}


### Assign labels
lookup= dict()
n_at_max = []
all_topics = []

for el in df['topics']:
    lookup= dict()

    if len(el) == 0:
        n_at_max.append(0)
        all_topics.append(pd.NA)
        leading_topic.append(pd.NA)
        continue

    for key in subtopics:
        lookup[key] = sum(x in el for x in subtopics[key])
    
    all_topics.append([k for k, v in lookup.items() if v == max(lookup.values())]) #Returns all items as list

    n_at_max.append(len([k for k, v in lookup.items() if v == max(lookup.values())]))
    
len(all_topics)
len(n_at_max)


selected_topics = []
for i in range(len(all_topics)):
    if n_at_max[i] == 0:
        selected_topics.append(pd.NA)
        continue

    if n_at_max[i] == 1:
        selected_topics.append(all_topics[i][0])
        continue

    if n_at_max[i] > 1:
        selected_topics.append(all_topics[i][random.randint(0,len(all_topics[i])-1)])

selected_topics_series = pd.Series(selected_topics)

fig = px.histogram(x=selected_topics_series.dropna() )
fig.show()

#Create labels
df['topic_label'] = selected_topics_series
#create labeled dataframe
df_labeled = df[df['topic_label'].notna()].reset_index(drop=True)

#reordering columns
df_labeled = df_labeled[['newid', 'date', 'topic_label', 'topics', 'places', 'people', 'orgs', 'exchanges','companies', 'title', 'dateline', 'text', 'author' ]]

#Adding Data
df_labeled.to_json("labeled_data.json")

