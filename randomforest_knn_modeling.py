
from wordprocessor import WordProcessor
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
import time
import plotly.express as px
import plotly.graph_objects as go

#I did a pretty poor job documenting what I was doing as I went through, looking at different models, etc.
def train_test():
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

    return train_x, train_y, test_x, test_y

train_x, train_y, test_x, test_y = train_test()

#I think the part of speech caused a few issues, but maybe not? SCIENCE.
    #After testing they don't add much value

#train_x_no_pos = train_x.drop(['CC','CD', 'DT','EX', 'FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP','RB','RBR','RBS','RP','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB', 'EXCEPT'], axis =1)
#test_x_no_pos = test_x.drop(['CC','CD', 'DT','EX', 'FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP','RB','RBR','RBS','RP','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB', 'EXCEPT'], axis =1)

"""
After testing the below parameters, parts of speech didn't do much.  They may work for a compacted model, however that is not our goal
"""
#train_x_no_NE = train_x.drop(total_corpus.entities, axis = 1)
#test_x_no_NE = test_x.drop(total_corpus.entities, axis = 1)
"""
Models with named entities out performed models without.  
"""

#Set up some trial parameters
param_grid = {
    'bootstrap': [True],
    'criterion': ['gini'],
    'max_depth': [None],
    'max_features': ['log2'],
    'min_samples_leaf': [1],
    'min_samples_split': [4, 8, 12],
    'n_estimators': [400, 600],
    'n_jobs' : [-1]
}
#First attempt was with no POS, second no _NE
def BruteForce( param_grid , compare = 'no_NE'):
    all_params = ParameterGrid(param_grid)
    all_times = np.array([])
    #Brute Force.
    results = {
        'model': [],
        'params': [],
        'f1': [],
        'accuracy': []
    }
    len(all_params)
    
    i = 0

    for param in all_params:
        lap = time.perf_counter()

        results['params'].append(param)
        results['model'].append('all')

        clf = RandomForestClassifier( **param)
        clf.fit(train_x,train_y)

        results['accuracy'].append(clf.score(test_x, test_y))
        test_preds = clf.predict(test_x)
        results['f1'].append(f1_score(test_y, test_preds , average = 'weighted'))

        if compare == 'no_pos':
            results['params'].append(param)
            results['model'].append('no_pos')
            clf_no_pos = RandomForestClassifier( **param)
            clf_no_pos.fit(train_x_no_pos,train_y)
            
            results['accuracy'].append(clf_no_pos.score(test_x_no_pos, test_y))
            test_preds_no_pos = clf_no_pos.predict(test_x_no_pos)

            results['f1'].append(f1_score(test_y, test_preds_no_pos , average = 'weighted'))

        if compare == 'no_NE':
            results['params'].append(param)
            results['model'].append('no_NE')
            clf_no_NE = RandomForestClassifier( **param)
            clf_no_NE.fit(train_x_no_NE,train_y)
            
            results['accuracy'].append(clf_no_NE.score(test_x_no_NE, test_y))
            test_preds_no_NE = clf_no_NE.predict(test_x_no_NE)

            results['f1'].append(f1_score(test_y, test_preds_no_NE , average = 'weighted'))


        
        timer = time.perf_counter()
        

        all_times = np.append(all_times, timer-lap)
        

        if i % 5 == 0:
            print('Completion percent: ', round((i/ len(all_params))*100, 3), '%','\n', 'est time rem: ', round((np.mean(all_times) * (len(all_params)-i))/60, 2), ' mins')
        i += 1
    #print(results)
    return pd.DataFrame(results)
    
#results = BruteForce(param_grid) #First Test
#results.to_json('candidate_models.json')

#This is just looking over model results
results = BruteForce(param_grid) #Second Test
results.to_json('candidate_models_2.json')
len(pd.read_json('candidate_models.json'))
models = pd.read_json('candidate_models.json')
models = models.loc[models['f1'] > .5]

params = pd.DataFrame(list(models['params']))
models = models.join(params)

models.drop('params', inplace= True, axis = 1)
models.drop('n_jobs', inplace= True, axis = 1)

top25 = models.sort_values('accuracy', ascending= False).head(25)
models['accuracy'].max()

px.scatter(top25, x= 'min_samples_split', y = 'f1', color = 'criterion').show()
px.histogram(models['f1'], title = 'F1 Scores of Parameter Testing').show()


models['min_samples_split'] = models['min_samples_split'].astype('category')
models['min_samples_leaf'] = models['min_samples_leaf'].astype('category')
models.loc[models['max_depth'].isna(), 'max_depth'] = 'N/A'

px.scatter(models, x= 'n_estimators', y= 'f1', color = 'min_samples_split').show()

models.sort_values('accuracy', ascending= False).head(10)


#Final Pass to test models
    #Pseudo-Cross Validation
    #Resample and build 6 models, we will average the model performance

all_dfs = []
param_grid = {
    'bootstrap': [True],
    'criterion': ['gini'],
    'max_depth': [None],
    'max_features': ['log2'],
    'min_samples_leaf': [1],
    'min_samples_split': [4, 8, 12],
    'n_estimators': [400, 600],
    'n_jobs' : [-1]
}
#Generate new training and test data 10 times, track models through process
for i in range(10):
    total_corpus = WordProcessor(full_data)
    total_corpus.get_all()
    ###
    total_corpus.data.drop('text', axis = 1, inplace= True)
    total_corpus.data.drop('date', axis = 1, inplace= True)

    train_y = total_corpus.data.loc[total_corpus.data.index.isin(total_corpus.train_rows)]['topic_label']
    train_x = total_corpus.data.loc[total_corpus.data.index.isin(total_corpus.train_rows)].drop('topic_label', axis = 1)
    test_y = total_corpus.data.loc[~total_corpus.data.index.isin(total_corpus.train_rows)]['topic_label']
    test_x = total_corpus.data.loc[~total_corpus.data.index.isin(total_corpus.train_rows)].drop('topic_label', axis = 1)
    

    results = BruteForce(param_grid, compare= 'none')

    all_dfs.append(results)
    
    print((1+i)/10, r' % done ',time.strftime("%H:%M:%S" ,time.localtime()))


for i in range(len(all_dfs)):
    all_dfs[i]['sample'] = i+1

#Sorting through Final results
model_eval = pd.concat(all_dfs)
params = pd.DataFrame(list(model_eval['params']))
model_eval = model_eval.join(params)
model_eval.drop('params', inplace= True, axis = 1)
model_eval.drop('n_jobs', inplace= True, axis = 1)
model_eval.drop('model', inplace= True, axis = 1)
model_eval.drop('bootstrap', inplace= True, axis = 1)
model_eval.drop('criterion', inplace= True, axis = 1)
model_eval.drop('max_depth', inplace= True, axis = 1)
model_eval.drop('max_features', inplace= True, axis = 1)
model_eval.drop('min_samples_leaf', inplace= True, axis = 1)

model_eval.reset_index(inplace = True, drop = True)

model_eval['sample'] = model_eval['sample'].astype(str)
model_eval['min_samples_split'] = model_eval['min_samples_split'].astype(str)
model_eval['n_estimators'] = model_eval['n_estimators'].astype(str)

model_eval.groupby(['min_samples_split', 'n_estimators']).agg(['mean', 'median', 'var'])['f1', 'var']**(.5)

perf_summary = model_eval.groupby(['min_samples_split', 'n_estimators']).agg(['mean', 'median'])
perf_summary2 = perf_summary.reset_index()
perf_summary2.columns = perf_summary2.columns.get_level_values(0)
perf_summary2.columns = ['min_samples_split', 'n_estimators', 'f1_mean', 'f1_median',  'accuracy_mean',  'accuracy_median']
perf_summary2.to_csv('perf_summary.csv', index = False)

px.scatter(model_eval, x= 'sample', y = 'f1', color= 'n_estimators', 
    facet_col = 'min_samples_split', title= 'Final Model Selection Over 10 Samples').show()
px.imshow()

perf_summary = pd.read_csv('perf_summary.csv')

perf_summary
#Summary Table
fig = go.Figure(data=[go.Table(
    header=dict(values=list(perf_summary.columns),
                align='left'),
    cells=dict(values=[perf_summary.min_samples_split, perf_summary.n_estimators, round(perf_summary.f1_mean,4), 
                    round(perf_summary.f1_median,4), round(perf_summary.accuracy_mean,4),  round(perf_summary.accuracy_median,4)],
                color = [] ,

               align='left'))
])
fig.show()



all_cms = []
for i in range(3):
    total_corpus = WordProcessor(full_data)
    total_corpus.get_all()
    total_corpus.data.drop('text', axis = 1, inplace= True)
    total_corpus.data.drop('date', axis = 1, inplace= True)

    train_y = total_corpus.data.loc[total_corpus.data.index.isin(total_corpus.train_rows)]['topic_label']
    train_x = total_corpus.data.loc[total_corpus.data.index.isin(total_corpus.train_rows)].drop('topic_label', axis = 1)
    test_y = total_corpus.data.loc[~total_corpus.data.index.isin(total_corpus.train_rows)]['topic_label']
    test_x = total_corpus.data.loc[~total_corpus.data.index.isin(total_corpus.train_rows)].drop('topic_label', axis = 1)
    
    clf = RandomForestClassifier( bootstrap= True, criterion= 'gini', max_depth = None, max_features= 'log2',  min_samples_leaf= 1, min_samples_split = 4, n_estimators = 600, n_jobs= -1 )

    clf.fit(train_x,train_y)
    test_preds = clf.predict(test_x)
    conf_mat = confusion_matrix(test_y, test_preds)

    all_cms.append(conf_mat)
    print((1+i)/10, r' % done ',time.strftime("%H:%M:%S" ,time.localtime()))

all_cms[0]
all_cms[1]

disp = ConfusionMatrixDisplay(np.mean(all_cms, axis = 0), display_labels = clf.classes_)
disp.plot(xticks_rotation = 'vertical')
plt.show()

clf.feature_importances_

all_cms[0]
all_cms[1]
all_cms[2]

fig = px.histogram(full_data, x= 'topic_label')
fig.show()



##### K Nearest Neighbors
    #Too much memory usage.
def BruteForce_KNN(param_grid, train_x, train_y, test_x, test_y):
    all_params = ParameterGrid(param_grid)
    all_times = np.array([])
    #Brute Force.
    results = {
        'params': [],
        'f1': [],
        'accuracy': []
    }
    
    i = 0

    for param in all_params:
        lap = time.perf_counter()

        results['params'].append(param)

        neigh = KNeighborsClassifier(**param)
        neigh.fit(train_x,train_y)
        test_preds = neigh.predict(test_x)

        results['accuracy'].append(accuracy_score(test_y, test_preds))
        results['f1'].append(f1_score(test_y, test_preds , average = 'weighted'))

        conf_mat = confusion_matrix(test_y, test_preds)

        timer = time.perf_counter()

        all_times = np.append(all_times, timer-lap)
        

        if i % 5 == 0:
            print('Completion percent: ', round((i/ len(all_params))*100, 3), '%','\n', 'est time rem: ', round((np.mean(all_times) * (len(all_params)-i))/60, 2), ' mins')
        i += 1
    #print(results)
    return pd.DataFrame(results), conf_mat


scale = StandardScaler().fit(train_x)
train_x_scaled = scale.transform(train_x)
test_x_scaled = scale.transform(test_x)


#Start with 50 neighbors
    #Accuray = .646
    #F1 = .637

param_grid = {
    'n_neighbors': list(range(5,110,25)),
    'algorithm' :['ball_tree'],
    'leaf_size' : [30],
    'n_jobs' : [-1]
}
len(ParameterGrid(param_grid))

kNN_tune_scale = BruteForce_KNN(param_grid, train_x_scaled, train_y, test_x_scaled, test_y)
kNN_tune = BruteForce_KNN(param_grid, train_x, train_y, test_x, test_y)

#kNN_tune_scale.to_json('knn_tune_scaled.json')
#kNN_tune.to_json('knn_tune.json')

kNN_tune_scale
kNN_tune
kNN_tune.loc[0,'params']
kNN_tune.loc[5, 'params']

neigh = KNeighborsClassifier(n_neighbors=100, algorithm='ball_tree', leaf_size= 100, n_jobs = -1)
neigh.fit(train_x, train_y)
test_preds = neigh.predict(test_x)
accuracy_score(test_y, test_preds )
f1_score(test_y, test_preds , average = 'weighted')

param_grid = {
    'n_neighbors': list(range(1,11)),
    'algorithm' :['ball_tree'],
    'leaf_size' : [30],
    'n_jobs' : [-1]
}
kNN_tune2 = BruteForce_KNN(param_grid, train_x, train_y, test_x, test_y)
kNN_tune2.to_json('knn_tune.json')
param_grid = {
    'n_neighbors': list(range(10,21)),
    'algorithm' :['ball_tree'],
    'leaf_size' : [30],
    'n_jobs' : [-1]
}
kNN_tune2 = BruteForce_KNN(param_grid, train_x, train_y, test_x, test_y)
kNN_tune2.to_json('knn_tune2.json')

kNN_tune = pd.read_json('knn_tune.json')


kNN_tune
kNN_tune2


param_grid = {
    'n_neighbors': list(range(1,4)),
    'algorithm' :['ball_tree'],
    'leaf_size' : [30],
    'n_jobs' : [-1]
}


for i in range(3):
    print(kNN_tune.loc[i,'params'])

all_cms = []
knn_final_selection = kNN_tune.loc[0:2].copy()
knn_final_selection

for i in range(10):
    train_x, train_y, test_x, test_y = train_test()

    df, conf_mat = BruteForce_KNN(param_grid)

    knn_final_selection.append(df, ignore_index=True)

    all_cms.append(conf_mat)

knn_final_selection.to_json('knn_tuned.json')

knn_final_selection





all_cms

neigh = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree', leaf_size= 50, n_jobs = -1)
neigh.fit(train_x, train_y)
test_preds = neigh.predict(test_x)
accuracy_score(test_y, test_preds )
f1_score(test_y, test_preds , average = 'weighted')

conf_mat = confusion_matrix(test_y, test_preds)

disp = ConfusionMatrixDisplay(conf_mat, display_labels = neigh.classes_)
disp.plot(xticks_rotation = 'vertical')


plt.show()


