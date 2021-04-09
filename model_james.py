from prototype_james import WordProcessor
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

full_data = pd.read_json('labeled_data.json', convert_dates= False)
#Cleaning Date
full_data['date'] = full_data['date'].str.slice(start = 0, stop = 20)
full_data['date'] = pd.to_datetime(full_data['date'])

total_corpus = WordProcessor(full_data)
total_corpus.get_all()

###
total_corpus.data.drop('text', axis = 1, inplace= True)
total_corpus.data.drop('date', axis = 1, inplace= True)


train_y = total_corpus.data.loc[total_corpus.data.index.isin(total_corpus.train_rows)]['topic_label']
train_x = total_corpus.data.loc[total_corpus.data.index.isin(total_corpus.train_rows)].drop('topic_label', axis = 1)
test_y = total_corpus.data.loc[~total_corpus.data.index.isin(total_corpus.train_rows)]['topic_label']
test_x = total_corpus.data.loc[~total_corpus.data.index.isin(total_corpus.train_rows)].drop('topic_label', axis = 1)


print(len(train_y))
print(len(train_x))
print(len(test_y))
print(len(test_x))

clf = RandomForestClassifier(n_estimators = 1001, n_jobs = -1, )
clf.fit(train_x,train_y)
clf.score(test_x, test_y)
clf.predict(test_x)


gbmclf = GradientBoostingClassifier(n_estimators = 100, learning_rate = 1)
gbmclf.fit(train_x,train_y)
gbmclf.score(test_x, test_y)

