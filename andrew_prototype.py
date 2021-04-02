import pandas as pd
import numpy as np
import re

#To run spacy you need to install both spacy and en_core_web_md model into your python enviroment
import spacy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

from matplotlib import pyplot as plt
import matplotlib


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

model = OneVsRestClassifier(LinearSVC(random_state=20258))

model.fit(vectorized_train_docs, train_labels)
predictions = model.predict(vectorized_test_docs)

def evaluate(test_labels, predictions):
    precision = precision_score(test_labels, predictions, average='micro', zero_division=0)
    recall = recall_score(test_labels, predictions, average='micro', zero_division=0)
    f1 = f1_score(test_labels, predictions, average='micro', zero_division=0)
    print("Micro-average fit scores")
    print("Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(precision, recall, f1))
    precision = precision_score(test_labels, predictions, average='macro', zero_division=0)
    recall = recall_score(test_labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(test_labels, predictions, average='macro', zero_division=0)
    print("Macro-average fit scores")
    print("Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(precision, recall, f1))
    
evaluate(test_labels, predictions)

#found a random earnings report for apple of the internet and our model classifies it what up!!
newtext = 'Cupertino, California — January 27, 2021 — Apple today announced financial results for its fiscal 2021 first quarter ended December 26, 2020. The Company posted all-time record revenue of $111.4 billion, up 21 percent year over year, and quarterly earnings per diluted share of $1.68, up 35 percent. International sales accounted for 64 percent of the quarter’s revenue. “Our December quarter business performance was fueled by double-digit growth in each product category, which drove all-time revenue records in each of our geographic segments and an all-time high for our installed base of active devices,” said Luca Maestri, Apple’s CFO. “These results helped us generate record operating cash flow of $38.8 billion. We also returned over $30 billion to shareholders during the quarter as we maintain our target of reaching a net cash neutral position over time.”'

vectorized_newtext = tfidf.transform([newtext])

predictnew = model.predict(vectorized_newtext)
labellist[np.where(predictnew[0] == 1)].tolist()

cm = confusion_matrix(test_labels.argmax(axis=1), predictions.argmax(axis=1))
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = labellist)
disp.plot(xticks_rotation="vertical")
plt.show()

