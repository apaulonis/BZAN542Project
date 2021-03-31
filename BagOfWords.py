CORPUS = [
'the sky is blue',
'sky is blue and sky is beautiful',
'the beautiful sky is so blue',
'i love blue cheese'
]
new_doc = ['loving this blue sky today']

# Bag of Words Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer
def bow_extractor(corpus, ngram_range=(1,1)):
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

bow_vectorizer, bow_features = bow_extractor(CORPUS)
features = bow_features.todense()
print(features)

new_doc_features = bow_vectorizer.transform(new_doc)
new_doc_features = new_doc_features.todense()
print(new_doc_features)

feature_names = bow_vectorizer.get_feature_names()
print(feature_names)

import pandas as pd
def display_features(features, feature_names):
    df = pd.DataFrame(data=features,
    columns=feature_names)
    print(df)
display_features(features, feature_names)

