import pandas as pd
import numpy as np
import re


#To run spacy you need to install both spacy and en_core_web_md model into your python enviroment
import spacy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_md", disable=["parser", "ner", "entity_linker", "entity_ruler", "textcat", "textcat_multilabel"])

df = pd.read_json('labeled_data.json')
df.info()

df.head()

train_docs, test_docs = train_test_split(df, test_size =0.25, random_state=14598)

lb = LabelBinarizer()
train_labels = lb.fit_transform([row['topic_label'] for i, row in train_docs.iterrows()])
test_labels = lb.transform([row['topic_label'] for i, row in test_docs.iterrows()])

labellist = lb.classes_

#does not check spelling have seen spelling errors adjusting for this could be a good future update
def tokenize(text):
    text = text.lower()
    min_length = 3
    tokens = [word.lemma_ for word in nlp(text) if not word.is_stop]
    p = re.compile('[a-zA-Z]+')
    filtered_tokens = list(filter (lambda token: p.match(token) and len(token) >= min_length, tokens))
    return filtered_tokens

#a quick check to make sure tokenize is working as we would like
train_docs.iloc[4, -2]
tokenize(train_docs.iloc[4,-2])

#this section is a hoss will take awhile do not be alarmed
tfidf = TfidfVectorizer(tokenizer=tokenize)
vectorized_train_docs = tfidf.fit_transform(train_docs.loc[:,'text'].to_list())
vectorized_test_docs = tfidf.transform(test_docs.loc[:,'text'].to_list())

vectorized_train_docs.shape

#a bit of a look into the sparse matrix this is big file so it has a lot
print(vectorized_train_docs[4,:])

#the actually word that is being checked in the above print out this was one of the words with a higher tfidf score
tokens = tfidf.get_feature_names()
tokens[5382]

