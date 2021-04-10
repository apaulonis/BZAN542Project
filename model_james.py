
from prototype_james import WordProcessor
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt  
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, plot_confusion_matrix, multilabel_confusion_matrix
from sklearn.model_selection import ParameterGrid

full_data = pd.read_json('labeled_data.json', convert_dates= False)

#Cleaning Date
full_data['date'] = full_data['date'].str.slice(start = 0, stop = 20)
full_data['date'] = pd.to_datetime(full_data['date'])

total_corpus = WordProcessor(full_data)
total_corpus.get_all()

###
total_corpus.data.drop('text', axis = 1, inplace= True)


train_y = total_corpus.data.loc[total_corpus.data.index.isin(total_corpus.train_rows)]['topic_label']
train_x = total_corpus.data.loc[total_corpus.data.index.isin(total_corpus.train_rows)].drop('topic_label', axis = 1)
test_y = total_corpus.data.loc[~total_corpus.data.index.isin(total_corpus.train_rows)]['topic_label']
test_x = total_corpus.data.loc[~total_corpus.data.index.isin(total_corpus.train_rows)].drop('topic_label', axis = 1)

#I think the part of speech caused a few issues, but maybe not? SCIENCE.
train_x_no_pos = train_x.drop(['CC','CD', 'DT','EX', 'FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP','RB','RBR','RBS','RP','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB', 'EXCEPT'], axis =1)
test_x_no_pos = test_x.drop(['CC','CD', 'DT','EX', 'FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP','RB','RBR','RBS','RP','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB', 'EXCEPT'], axis =1)

#Set up some trail parameters
param_grid = {
    'bootstrap': [True, False],
    'max_depth': [None, 100, 500],
    'max_features': ['auto', .33, 'log2'],
    'min_samples_leaf': [1, 3, 5],
    'min_samples_split': [2, 10, 15],
    'n_estimators': [50, 100, 200, 300],
    'n_jobs' : [-1]
}

#Brute Force.
results = {
    'model': [],
    'params': [],
    'f1': [],
    'precision': [],
    'recall': [],
    'accuracy': []
}
all_params = ParameterGrid(param_grid)
len(all_params)
i = 0
for param in all_params:
    print(param)
    results['params'].append(param)
    results['model'].append('all')

    clf = RandomForestClassifier( **param)
    clf.fit(train_x,train_y)

    results['accuracy'].append(clf.score(test_x, test_y))
    test_preds = clf.predict(test_x)
    results['f1'].append(f1_score(test_y, test_preds , average = 'weighted'))
    results['precision'].append(precision_score(test_y, test_preds , average = 'weighted'))
    results['recall'].append(recall_score(test_y, test_preds , average = 'weighted'))

    results['params'].append(param)
    results['model'].append('no_pos')

    clf_no_pos = RandomForestClassifier( **param)
    clf_no_pos.fit(train_x_no_pos,train_y)
    
    results['accuracy'].append(clf_no_pos.score(test_x_no_pos, test_y))
    test_preds_no_pos = clf_no_pos.predict(test_x_no_pos)

    results['f1'].append(f1_score(test_y, test_preds_no_pos , average = 'weighted'))
    results['precision'].append(precision_score(test_y, test_preds_no_pos , average = 'weighted'))
    results['recall'].append(recall_score(test_y, test_preds_no_pos , average = 'weighted'))

    i += 1

    print('Completion percent: ', 1/ len(all_params))

