'''
This script performs multi-class classification of text using two forms of feature extraction,
TF-IDF from scikit-learn and embedded word vectors from spaCy.  The classification modeling 
between the created features and the text labels is handled by LinearSVC from scikit-learn.

This script handles multi-class, but not multi-label, text classification.  Scikit-learn
functions are capable of multi-label classification, but it was not attempted here.  This
code requires a single label per record as created in the feature engineering script in 
this repo. Extending this code to the multi-label problem would be a logical next step.

Andrew Paulonis, July 2021
'''
import pandas as pd
import numpy as np
import re
import warnings

# Note that to run spaCy you need to install both spaCy and the en_core_web_md pipeline
# into your python environment.  This requirement is noted in the readme.
import spacy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from sklearn.exceptions import ConvergenceWarning

from scipy.stats import loguniform

from matplotlib import pyplot as plt
import matplotlib

# Load spaCy and the en_core_web_md pipeline.  This pipline accepts English text and includes a
# word vector model with 685k keys, 20k unique vectors, and 300 dimensions.  There is a
# pipline with a much larger model including 685k unique vectors, but the smaller model
# works fine for this project.
#
# Also disable many of the spacy functions.  We mostly need the word vector model for this
# project.  Disabling unused functions speeds things up and uses less memory.
nlp = spacy.load("en_core_web_md",
                 disable=["parser", "ner", "entity_linker", 
                          "entity_ruler", "textcat", "textcat_multilabel"])

# Read the cleaned text records with a single label per record
df = pd.read_json('labeled_data.json', convert_dates=False)

# Display some information about the input data
print('\nInformation about the labeled data\n')
df.info()
df.head()

# Train on a random set of 75% of the records.  Hold back 25% for testing.
train_docs, test_docs = train_test_split(df, test_size=0.25, random_state=14598)

# For the multi-label problem, use the LabelBinarizer to convert the multi classification
# problem into N binary classification problems by creating N binary (0/1) label vectors
# for each value in the text label set.
lb = LabelBinarizer()

# Use fit_transform to create the initial binarization with the training data.
train_labels = lb.fit_transform([row['topic_label'] for i, row in train_docs.iterrows()])

# Use transform only to binarize the test data using the model from the training data.
# Note that if there are any labels in the test set that are not in the training set, the
# function will generate a warning.  Since we do not have any rare labels, this was not
# problem.  If it was a problem, using stratified sampling for train-test split would
# be important.
test_labels = lb.transform([row['topic_label'] for i, row in test_docs.iterrows()])

# Save a list of the labels for future use
labellist = lb.classes_
print('\nList of text labels\n')
print(labellist)

# Define a custom tokenization function for the TfidfVectorizer from scikit-learn.
# A default tokenizer is available, but since we have loaded spaCy, we can use
# the spaCy stopwords and lemmatizer, which should be as good or better than
# what is in scikit-learn
#
# Note that this tokenizer does not check spelling. This could be a good addition
# in a future update.
#
# This tokenizer also returns only unigrams (1-grams).  There is some possibility
# that including bigrams or trigrams could improve model performance even further.
#
# Tokenizer steps:
# - convert all words to lower case
# - do not include words less than 3 characters
# - filter out default spaCy stopword set
# - replace all words with their spaCy lemma
# - remove words with numbers
def tokenize(text):
    text = text.lower()
    min_length = 3
    tokens = [word.lemma_ for word in nlp(text) if not word.is_stop]
    p = re.compile('[a-zA-Z]+')
    filtered_tokens = list(filter (lambda token: p.match(token) and len(token) >= min_length, tokens))
    return filtered_tokens
  
# Add stopword "reuter" to spaCy default list - this word shows up in most documents
nlp.Defaults.stop_words.add("reuter")

# Do a quick check to make sure tokenize is working as we would like.
# Print the text of one of the records 
print('\nText from example record\n')
print(train_docs.iloc[5, -2])
# Print the list of tokens generated from that text
print('\nTokens from that text\n')
print(tokenize(train_docs.iloc[5,-2]))

# Convert the text in all the documents into a TF-IDF matrix.
# This section requires a lot of computation so it will take a while. Do not be alarmed.
print('\nCreating the TF-IDF matrix. This takes a while - do not be alarmed\n')
tfidf = TfidfVectorizer(tokenizer=tokenize)
# Create the TF-IDF model on the training documents using fit_transform
vectorized_train_docs = tfidf.fit_transform(train_docs.loc[:,'text'].to_list())
# Use that trained model on the test documents with transform only
vectorized_test_docs = tfidf.transform(test_docs.loc[:,'text'].to_list())

# Display the shape of the TF-IDF matrix to get an idea of how big it is.
# There is one row for each document (training in this case), and a column for 
# each token.  It is a sparse matrix which helps a lot since it is so big, but
# has a lot of zeros (most documents have only a few tokens out of the full set).
print('\nSize of training TF-IDF matrix\n')
print(vectorized_train_docs.shape)

# What do the TF-IDF values look like for an example document
print('\nTF-IDF row for the example record\n')
# Get the sparse row for the example document in the training set and convert to a
# dense array
row = vectorized_train_docs[5,:].toarray()
# Convert the array to a dataframe and transpose it for easier display
df_row = pd.DataFrame(row).transpose()
# Add a new column with the token names corresponding to the tf-idf scores
df_row['token'] = tfidf.get_feature_names()
# Rename the tf-idf column
df_row.rename(columns={0:'tf-idf'}, inplace=True)
# Filter out the huge number of tokens with zero tf-idf
df_row = df_row[df_row['tf-idf'] != 0]
print(df_row)

# Use the one vs rest strategy for multi-class classification.
# Use the LinearSVC algorithm to do the modeling.
# Start with all the defaults for LinearSVC.
model = OneVsRestClassifier(LinearSVC(random_state=20258))

# Fit a support vector classifier model between the training TF-IDF values
# and the training labels.
model.fit(vectorized_train_docs, train_labels)

# Create predictions of the test set labels by passing the test TF-IDF values
# through the SVC model.
predictions = model.predict(vectorized_test_docs)

# Create a function to compute several classification metrics on the test data.
# This will get used again for the modeling based on spaCy features.
def evaluate(test_labels, predictions):
    # Precision, recall, and F1 give a good overall picture of model performance
    # Compute metrics using 'micro' and 'macro' averaging.  Micro averaging gives
    # a more meaningful answer when label count is imbalanced across the documents.
    # None of the labels are too rare in this dataset, so micro and macro averages
    # are pretty close to the same.
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
    
# Compute and print the metrics
print('\nFit metrics for the TF-IDF/SVC model\n')
evaluate(test_labels, predictions)

# Test the predictions with some completely new text from a recent earnings report.
# One of the class labels for the Reuters news text was earnings, so the model
# should be able to classify text from an earnings reports as earnings.
newtext = ('Cupertino, California — January 27, 2021 — Apple today announced financial '
  'results for its fiscal 2021 first quarter ended December 26, 2020. The Company posted '
  'all-time record revenue of $111.4 billion, up 21 percent year over year, and quarterly ' 
  'earnings per diluted share of $1.68, up 35 percent. International sales accounted for ' 
  '64 percent of the quarter’s revenue. “Our December quarter business performance was '
  'fueled by double-digit growth in each product category, which drove all-time revenue '
  'records in each of our geographic segments and an all-time high for our installed base '
  'of active devices,” said Luca Maestri, Apple’s CFO. “These results helped us generate '
  'record operating cash flow of $38.8 billion. We also returned over $30 billion to '
  'shareholders during the quarter as we maintain our target of reaching a net cash neutral '
  'position over time.”')

# Compute the TF-IDF values for the new test text.
vectorized_newtext = tfidf.transform([newtext])

# Predict the label using the Reuters model
predictnew = model.predict(vectorized_newtext)

# Convert the prediction vector to the readable label
newlabel = labellist[np.where(predictnew[0] == 1)].tolist()

# Show the results - which happen to be correct
print('\nTesting the model with text from a recent earnings report\n')
print(newtext)
print('\nThe predicted label is:')
print(newlabel)

# Create and display a confusion matrix for the test prediction results
# The label 'acquisitions' is falsely predicted the most.  Other labels have
# very few false negatives or positives.  This great performance would be 
# expected given the high value of 0.95 for micro-average F1.
cm = confusion_matrix(test_labels.argmax(axis=1), predictions.argmax(axis=1))
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = labellist)
disp.plot(xticks_rotation="vertical")
plt.show()

# The model thus far was trained using the default parameters for LinearSVC.
# Try to optimize the model by changing some of the parameters.  LinearSVC does
# not have a lot of parameters to adjust, but there are some.
#
# Create a dictionary with parameter options.  The regularization parameter C
# is the most obvious parameter to adjust.  It has a wide possible range, so
# make it a loguniform distribution.  For a couple other parameters, choose
# between discrete options.
params = {'estimator__C': loguniform(0.01, 10),
          'estimator__class_weight': [None,'balanced'],
          'estimator__loss': ['squared_hinge','hinge']}

print('\nFitting an optimized SVC model - short delay\n')
# Instantiate a new model to be optimized
model_tune = OneVsRestClassifier(LinearSVC(max_iter=1000, random_state=1000))

# Use a randomized search over the parameters.  Use 5-fold cross-validation and
# 10 iterations.  Score on the F1 micro-averaged metric.
search = RandomizedSearchCV(estimator=model_tune, param_distributions=params, 
  scoring='f1_micro', n_iter=10, cv=5, n_jobs=1, verbose=0, random_state=1000)
# Even with 1000 iterations, there will still be convergence warnings when
# trying to fit SVC with some parameter combinations.  Suppress these warnings.
# You may want to remove the suppression if experimenting with fitting or
# optimization.
with warnings.catch_warnings():
  warnings.filterwarnings("ignore", 
                          category=ConvergenceWarning,
                          module="sklearn")
  search.fit(vectorized_train_docs, train_labels)

print('Best params: ',search.best_params_)
print('\nBest Cross Validation Score: {:.4f}'.format(search.best_score_))

# Predict the test set with the optimized model and show the performance.
predictions_tune = search.predict(vectorized_test_docs)
print('\nFit metrics for the optimized TF-IDF/SVC model\n')
evaluate(test_labels, predictions_tune)

# Note that tuning did not make much difference in F1 micro-averaged score.
# The default options worked well in this case.

# Next we will attempt to use a SVC with a language model doc2vec approach using spaCy.
# This will turn each document into a 300 dimensional vector based on the 
# spaCy model of the english language
#
# First tokenize the text to unigrams of 3 characters or more that do not contain numbers.
# Remove stopword tokens.  Then find the 300-dimensional vector representation
# of each token using the language model in the spaCy en_core_web_md pipeline.
# Compute a vector representation of the full text by averaging the vectors 
# of all the tokens in the text.
def doc2vec(text):
    min_length = 3
    p = re.compile('[a-zA-Z]+')
    tokens = [token for token in nlp(text) if not token.is_stop and
              p.match(token.text) and
              len(token.text) >= min_length]
    doc = np.average([token.vector for token in tokens], axis=0)
    return doc 

# Apply the doc2vec function just defined to both the training and test set documents.
# These steps take a long time to finish, so do not panic when nothing seems to happen
# for a while.

print('\nCreating the word vector matrix with spaCy. This takes a while - do not be alarmed\n')
doc2vec_train_docs = np.array([doc2vec(doc) for doc in train_docs.loc[:,'text'].to_list()]) 
doc2vec_test_docs = np.array([doc2vec(doc) for doc in test_docs.loc[:,'text'].to_list()])

# Display the shape of the word vector matrix to get an idea of how big it is.
# There is one row for each document (training in this case), and a column for 
# each dimension of the language model.  This is not a sparse matrix.
print('Size of training word vector matrix\n')
print(doc2vec_train_docs.shape)

# Look into the word vector matrix for the example document
print('\nFirst 20 dimensions of the word vector row for the example record\n')
print(doc2vec_train_docs[4,:20])

# Time to build a SVC model again, this time with our doc2vec x matrix.
# Use the same approach as with the TF-IDF features, but in this case the
# doc2vec features are used.
model2 = OneVsRestClassifier(LinearSVC(random_state=1002))
model2.fit(doc2vec_train_docs, train_labels)
predictions2 = model2.predict(doc2vec_test_docs)
print('\nFit metrics for the doc2vec/SVC model\n')
evaluate(test_labels, predictions2)

# Now tune the doc2vec model using the same process as earlier with randomized
# search and cross-validation
params2 = {'estimator__C': loguniform(0.01, 100),
          'estimator__class_weight': [None,'balanced'],
          'estimator__loss': ['squared_hinge','hinge']}

print('\nFitting an optimized SVC model - short delay\n')
model2_tune = OneVsRestClassifier(LinearSVC(max_iter=1000, random_state=1002))
search2 = RandomizedSearchCV(estimator=model2_tune, param_distributions=params2, 
  scoring='f1_micro', n_iter=15, cv=5, n_jobs=1, verbose=0, random_state=1003)
# Even with 1000 iterations, there will still be convergence warnings when
# trying to fit SVC with some parameter combinations.  Suppress these warnings.
# You may want to remove the suppression if experimenting with fitting or
# optimization.
with warnings.catch_warnings():
  warnings.filterwarnings("ignore", 
                          category=ConvergenceWarning,
                          module="sklearn")
  search2.fit(doc2vec_train_docs, train_labels)

print('Best params: ',search2.best_params_)
print('\nBest Cross Validation Score: {:.4f}'.format(search2.best_score_))

# Predict the test set with the optimized model and show the performance.
predictions2_tune = search2.predict(doc2vec_test_docs)
print('\nFit metrics for the optimized doc2vec/SVC model\n')
evaluate(test_labels, predictions2_tune)

# Similar to the model from TF-IDF values, tuning did not make much difference
# in model performance.  We can use the model with default parameters.

# Create and display a confusion matrix for the test prediction results
# The label 'acquisitions' is falsely predicted the most.  Other labels have
# very few false negatives or positives.  This great performance would be 
# expected given the high value of 0.95 for micro-average F1.
cm = confusion_matrix(test_labels.argmax(axis=1), predictions2.argmax(axis=1))
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = labellist)
disp.plot(xticks_rotation="vertical")
plt.show()

# See how the model based on doc2vec does on the new text from the earnings report.
# Create the x values with doc2vec.
doc2vec_newtext = np.array([doc2vec(newtext)])
# Run those values through the model to predict the label.
predictD2Vnew = model2.predict(doc2vec_newtext)
newlabel2 = labellist[np.where(predictD2Vnew[0] == 1)].tolist()

# Show the results - which happen to be correct
print('\nTesting the doc2vec model with the earnings report text')
print('\nThe predicted label is:')
print(newlabel2)

# The doc2vec model also predicts the correct label for the new earnings text.
#
# Overall, the model based on TF-IDF features has higher performance based on F1
# score and the counts in the confusion matrix for the held back test set of the
# Reuters documents.
#
# However, the more generic language approach of the word vector model compared to the
# TF-IDF model that is very specific to the Reuters document set might make the
# word vector model predict better on text that did not come from Reuters.
# This hypothesis was not analyzed.