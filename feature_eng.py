'''
Create a single topic for each Reuters document based on the possibly multiple
topics in the raw data set.  The file cat-descriptions_120396.txt that 
included with the raw data files was used to establish an aggregate subtopic label
in six broad categories.  Information from the file as well as manual review of
topic labels in the data set were used to map actual labels to aggregate labels.
'''

import pandas as pd
import random
import plotly.express as px

# Set a random seed so any random tie-breaking below will be repeatable
random.seed(1111)

# Read the structured data that was parsed out of the raw files.
df = pd.read_json('clean_data.json', convert_dates=False)
# Display some information about the resulting dataframe
df.info()

# List labels from the raw data that map to six aggregate label subtopics
s_money_forex = {"MONEY-FX", "SHIP", "INTEREST", "DLR", "AUSTDLR", "HK", "SINGDLR",
                 "NZDLR", "CAN", "STG", "DMK", "YEN", "SFR", "FFR", "BFR", "DFL",
                 "LIT", "DKR", "NKR", "SKR", "MEXPESO", "CRUZADO", "AUSTRAL",
                 "SAUDRIYAL", "RAND", "RUPIAH", "RINGGIT", "ESCUDO", "PESETA", "DRACHMA"}
s_economic_indicator = {"BOP", "TRADE", "CPI", "WPI", "JOBS", "IPI", "CPU", "GNP",
                        "MONEY-SUPPLY", "RESERVES", "LEI", "HOUSING", "INCOME",
                        "INVENTORIES", "INSTAL-DEBT", "RETAIL"}
s_acquisitions = {"ACQ"}
s_earnings = {"EARN"}
s_commodity = {"ALUM", "BARLEY", "CARCASS", "CASTOR-MEAL", "CASTOR-OIL", "CASTORSEED",
               "CITRUSPULP", "COCOA", "COCONUT-OIL", "COCONUT", "COFFEE", "COPPER",
               "COPRA-CAKE", "CORN-OIL", "CORN", "CORNGLUTENFEED", "COTTON",
               "COTTON-MEAL", "COTTON-OIL", "COTTONSEED", "F-CATTLE", "FISHMEAL",
               "FLAXSEED", "GOLD", "GRAIN", "GROUNDNUT", "GROUNDNUT-MEAL",
               "GROUNDNUT-OIL", "IRON-STEEL", "LEAD", "LIN-MEAL", "LIN-OIL", "LINSEED",
               "LIVESTOCK", "L-CATTLE", "HOG", "LUMBER", "LUPIN", "MEAL-FEED",
               "NICKEL", "OAT", "OILSEED", "ORANGE", "PALLADIUM", "PALM-MEAL",
               "PALM-OIL", "PALMKERNEL", "PLATINUM", "PLYWOOD", "PORK-BELLY", "POTATO",
               "RAPE-MEAL", "RAPE-OIL", "RAPESEED", "RED-BEAN", "RICE", "RUBBER",
               "RYE", "SILK", "SILVER", "SORGHUM", "SOY-MEAL", "SOY-OIL", "SOYBEAN",
               "STRATEGIC-METAL", "SUGAR", "SUN-MEAL", "SUN-OIL", "SUNSEED", "TAPIOCA",
               "TEA", "TIN", "TUNG-OIL", "TUNG", "VEG-OIL", "WHEAT", "WOOL", "ZINC"}
s_energy = {"CRUDE", "HEAT", "FUEL", "GAS", "NAT-GAS", "PET-CHEM", "PROPANE", "JET", "NAPHTHA"}

# Convert all labels to lower case, since the text tokens will be in lower case
s_money_forex = {x.lower() for x in s_money_forex}
s_economic_indicator = {x.lower() for x in s_economic_indicator}
s_acquisitions = {x.lower() for x in s_acquisitions}
s_earnings = {x.lower() for x in s_earnings}
s_commodity = {x.lower() for x in s_commodity}
s_energy = {x.lower() for x in s_energy}

# Make dictionaries mapping the labels from the documents to an aggregate label
subtopics = {'money': s_money_forex, 
            'econ': s_economic_indicator, 
            'acquisitions': s_acquisitions,
            'earnings' : s_earnings,
            'commodity': s_commodity,
            'energy': s_energy}

# Display the dictionaries
print('\nAggregated subtopics\n')
print(subtopics)

# Find out which subtopic has the most matches with the raw topics for each document

n_at_max = []
all_topics = []

# Loop through each document in the data set
for el in df['topics']:
    lookup = dict()

    # If there are no topics for the document, then N=0 and topic = NA
    if len(el) == 0:
        n_at_max.append(0)
        all_topics.append(pd.NA)
        continue
    # If there are topics, find how many matches there are between the document
    # topic words and the words that represent each aggregate subtopic
    for key in subtopics:
        lookup[key] = sum(x in el for x in subtopics[key])
    # Find the subtopic(s) with highest match count and append a list of one or more to the main list
    all_topics.append([k for k, v in lookup.items() if v == max(lookup.values())])
    # Count how many subtopics are in the best match list.  Usually will be 1,
    # but will be more than 1 some times
    n_at_max.append(len([k for k, v in lookup.items() if v == max(lookup.values())]))

# Sanity check on the length of the lists just created.  They should be as long
# as the number of documents in the data set
print('\nLength of all_topics and n_at_max lists\n')
print(len(all_topics))
print(len(n_at_max))

# Check how many documents have more than one subtopic tied for the maximum
# number of matches
df_n = pd.DataFrame(n_at_max, columns=['N'])
df_n['count'] = 1
print('\nNumber of documents with N subtopic matches\n')
print(df_n.groupby(['N']).sum())

# Reduce to just one subtopic per document.  If there were no topics associated
# with the raw document, record NA for the subtopic.  If there was one subtopic
# that had the maximum match count, record that subtopic.  If there was more
# than one subtopic with the same maximum match count, choose randomly between
# the options and record the that subtopic
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
# Convert the selected topics list to a series so it can be appended to the
# main dataframe
selected_topics_series = pd.Series(selected_topics)

# Show the distribution of subtopics for documents containing a subtopic
fig = px.histogram(x=selected_topics_series.dropna())
fig.show()

# Add new subtopic labels to the main dataframe
df['topic_label'] = selected_topics_series
# Remove all documents that do not have a label
df_labeled = df[df['topic_label'].notna()].reset_index(drop=True)

# Reorder the dataframe columns
df_labeled = df_labeled[['newid', 'date', 'topic_label', 'topics', 'places', 'people', 'orgs', 'exchanges','companies', 'title', 'dateline', 'text', 'author' ]]

# Show some info on the labeled data
df_labeled.info()

# Write the labeled data out to a json file
df_labeled.to_json("labeled_data.json")

