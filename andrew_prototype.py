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
test_labels = lb.fit_transform([row['topic_label'] for i, row in test_docs.iterrows()])

labellist = lb.classes_

