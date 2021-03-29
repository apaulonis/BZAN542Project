


import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
import plotly.express as px
from collections import Counter


full_data = pd.read_json('labeled_data.json')

full_data.info()

#Stop words from nltk, punctuation from string
stop_words = set(stopwords.words('english'))
punc = {i for i in string.punctuation}
punc = punc.union({'--', '``', "''"})


class WordProcessor:

    def __init__(self, words) -> None:
        super().__init__()

        self.data_structure = type(words)

        self.all_words = words

        if isinstance(words, pd.core.frame.DataFrame ):
            self.all_words = words['text']



    def tokenizer(self, by = 'words'):
        """
        by = words, sentence are the only arguments built
        
        generates a self.tokenized attribute
        """
        if by == 'words':
            self.tokenized = nltk.tokenize.word_tokenize( self.all_words )
        
        if by == 'sentence':
            self.tokenized = nltk.tokenize.sent_tokenize( self.all_words )

    def clean(self, remove_stop_words = True, remove_punctuation = True, return_tokens = True):
        
        if remove_stop_words and remove_punctuation == False:
            stop_words = set(stopwords.words('english'))
            clean_words = [word.lower() for word in all_words if word not in stop_words]

        if remove_punctuation and remove_stop_words == False:
            punc = {i for i in string.punctuation}
            clean_words = [word.lower() for word in all_words if word not in punc]

        if remove_stop_words and remove_punctuation:
            stop_words = set(stopwords.words('english'))
            punc = {i for i in string.punctuation}

            clean_words = [word.lower() for word in all_words if word not in stop_words and word not in punc]

        if return_tokens:
            self.clean_words = clean_words
        else:
            self.clean_words = TreebankWordDetokenizer().detokenize(clean_words)

    def parts_of_speech():
        pos_dict = dict()
        for i in range(len(clean_tagged)):
            if clean_tagged[i][1] not in pos_dict.keys():
                pos_dict[clean_tagged[i][1]] = 1
                continue

        pos_dict[clean_tagged[i][1]] += 1


        pos_proportions = dict()
        for k, v in pos_dict.items():
            pos_proportions[k] = v/len(clean_words)




isinstance(WordProcessor(full_data).all_words, pd.core.frame.DataFrame )
WordProcessor(full_data).all_words



type(full_data)
type(full_data['text'])
type(full_data['text'][0])



#One element Stuff

#Word tokenization in order to remove punctuation and stop words
all_words = nltk.tokenize.word_tokenize(full_data['text'][0] )


clean_words = [word.lower() for word in all_words if word not in stop_words and word not in punc]
bigrams = list(nltk.bigrams(clean_words))
cfd = nltk.ConditionalFreqDist( bigrams)

clean_tagged = nltk.pos_tag(clean_words)

pos_dict = dict()
for i in range(len(clean_tagged)):
    if clean_tagged[i][1] not in pos_dict.keys():
        pos_dict[clean_tagged[i][1]] = 1
        continue

    pos_dict[clean_tagged[i][1]] += 1

pos_proportions = dict()
for k, v in pos_dict.items():
    pos_proportions[k] = v/len(clean_words)

pos_proportions

print(clean_tagged)
len(clean_tagged)
#Likely want to keep both lengths as a feature
len(all_words)
len(clean_words)


#Experimentation
print(list(cfd))

cfd['bahia']
TreebankWordDetokenizer().detokenize(clean_words)



##Series Wide stuff
clean_words = []
for i in range(len(full_data['text'])):
    all_words = nltk.tokenize.word_tokenize(full_data['text'][i] )
    clean_words.append([word.lower() for word in all_words if word.lower() not in stop_words and word.lower() not in punc])

## No words common to EVERY article
set_clean_words = set(clean_words[0])
for s in clean_words[1:]:
    set_clean_words.intersection_update(s)
list(set_clean_words)

# Adding a column of the clean words, tokenized by word
full_data['clean_text'] = clean_words

#Part of speech tagging
all_clean_tagged = []
for i in range(len(full_data['clean_text'])):
    clean_tagged = nltk.pos_tag(full_data['clean_text'][i])
    all_clean_tagged.append(clean_tagged)

    if i % 1000 == 0:
        print(i)
    
full_data['clean_tagged'] = all_clean_tagged

# Stemming.  First pass using stems, maybe lemmatize later
all_stemmed = []
porter_stem = nltk.PorterStemmer()
for i in range(len(full_data['clean_text'])):
    
    clean_stemmed =  [porter_stem.stem(word) for word in full_data['clean_text'][i]]

    all_stemmed.append(clean_stemmed)

    if i % 1000 == 0:
        print(i)

full_data['clean_stemmed'] = all_stemmed

#tagging stemmed version
all_clean_stemmed_tagged = []
for i in range(len(full_data['clean_text'])):
    clean_stemmed_tagged = nltk.pos_tag(full_data['clean_stemmed'][i])
    all_clean_stemmed_tagged.append(clean_stemmed_tagged)

    if i % 1000 == 0:
        print(i)
    
full_data['clean_stemmed_tagged'] = all_clean_stemmed_tagged


#Part of speech summary.
#All tags NLK may return
pos_comprehensive = ['CC','CD', 'DT','EX', 'FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP','RB','RBR','RBS','RP','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB', 'EXCEPT']

pos_counter = {key: 0 for key in pos_comprehensive}

#missing_keys = {'document':[], 'element':[], 'string':[]}
all_counts = []
all_proportions = []

for i in range(len(full_data['clean_tagged'])):
    pos_dict = pos_counter.copy()

    for j in range(len(full_data['clean_tagged'][i])):

        try:
            pos_dict[full_data['clean_tagged'][i][j][1]] += 1

        except:
            #missing_keys['document'].append(i)
            #missing_keys['element'].append(j)
            #missing_keys['string'].append(full_data['clean_tagged'][i][j][0])
            pos_dict['EXCEPT'] +=1 
    all_counts.append(pos_dict)

    if i % 1000 == 0:
        print(i)
        

    pos_proportions = dict()
    for k, v in pos_dict.items():
        pos_proportions[k] = v/len(full_data['clean_tagged'][i])

    all_proportions.append(pos_proportions)

    if i % 1000 == 0:
        print(i)


#### PART OF SPEECH STUFF WITH CLEAN, STEMMED AND TAGGED DATA
all_counts_stem = []
for i in range(len(full_data['clean_stemmed_tagged'])):
    pos_dict = pos_counter.copy()

    for j in range(len(full_data['clean_stemmed_tagged'][i])):

        try:
            pos_dict[full_data['clean_stemmed_tagged'][i][j][1]] += 1

        except:
            #missing_keys['document'].append(i)
            #missing_keys['element'].append(j)
            #missing_keys['string'].append(full_data['clean_tagged'][i][j][0])
            pos_dict['EXCEPT'] +=1 

    all_counts_stem.append(pos_dict)
    if i % 1000 == 0:
        print(i)



df_pos = pd.DataFrame(all_counts)
df_pos_prop = pd.DataFrame(all_proportions)

df_pos
df_pos_prop


#df_pos_stemmed = pd.DataFrame(all_counts_stem)

df_pos.head()
#df_pos_stemmed.head()

df_pos['EXCEPT'].sum()
#df_pos_stemmed['EXCEPT'].sum()


df_pos.apply(sum, axis= 0)




full_data = full_data.join(df_pos_prop)

full_data.columns

full_data = full_data.join(df_pos_prop, rsuffix= '_prob' )


cols = [x + '_prob' for x in 
['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD',
       'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'RB', 'RBR', 'RBS',
       'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP',
      'WP$', 'WRB', 'EXCEPT'] ]

ls= ['topic_label']
[ls.append(x) for x in cols]

ls
len(ls)
#plot_averages = full_data[['topic_label','CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD',
#       'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'RB', 'RBR', 'RBS',
#       'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP',
#       'WP$', 'WRB', 'EXCEPT']].groupby('topic_label', as_index= False).aggregate('mean')

"""
plot_averages.columns
fig = px.bar(
    plot_averages,
       x='topic_label',

       y= ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD',
       'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'RB', 'RBR', 'RBS',
       'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP',
       'WP$', 'WRB', 'EXCEPT' ]
)

fig.show()
"""
full_data.columns

plots = dict()



from plotly import tools
from plotly import subplots
import plotly.graph_objs as go


plots['trace' + str(0)] = 1

fig = subplots.make_subplots(rows = 18, cols = 2, subplot_titles= ls[1:])

for i, col in enumerate(ls[1:]):
    row = int(i/2) + 1
    col = i % 2 + 1
    print(row,col)

    fig_internal = go.Figure()
    for el in full_data['topic_label'].unique():
        fig.add_trace(go.Histogram(
            x= list(full_data[ls[i]][full_data['topic_label'] == el])),
            row = row, col = col)

    fig.update_layout(barmode = 'overlay')
    fig.update_traces(opacity = 0.4)

fig.update_layout(height = 5000, width = 2000)
fig.show()
int(35/2)+1 


2 // 2 +1


5//2
len(ls[1:])

full_data[ls[2]][full_data['topic_label'] == 'commodity']


fig2 = px.histogram(full_data[ls],
    x = full_data[ls[2]],
    opacity= 0.7,
    color = full_data['topic_label']  )

fig2.show()

fig3 = go.Histogram(
    x= list(full_data[ls[2]])

)
fig3.show()
['topics_list'].append(cols)

full_data.columns



#Generate a set of every word. Will iterate over to get a count of all words
every_word = set()
for i in range(len(full_data['clean_stemmed'])):
    every_word.update(full_data['clean_stemmed'][i] )

word_count_comprehensive = dict()
#Generate keys for a comprehensive word count
for key in every_word:
    word_count_comprehensive[key] = 0 

#Generate comprehensive word count
for i in range(len(full_data['clean_stemmed'])):
    for j in range(len(full_data['clean_stemmed'][i])):

        word_count_comprehensive[full_data['clean_stemmed'][i][j]] += 1


dict(sorted(word_count_comprehensive.items(), key=lambda item: item[1], reverse= True))


word_count2 = Counter()

for i in range(len(full_data['clean_stemmed'])):
    for j in range(len(full_data['clean_stemmed'][i])):

        word_count2[full_data['clean_stemmed'][i][j]] += 1

len(every_word)

top500_words = {tup[0] for tup in word_count2.most_common(500)}
top500_counts = dict()
top500_tally = dict()

for key in top500_words:
    top500_counts[key] = [] 
    top500_tally[key] = 0


for i in range(len(full_data['clean_stemmed'])):
    tally = top500_tally.copy()
    for j in range(len(full_data['clean_stemmed'][i])):
        
        try:
            tally[full_data['clean_stemmed'][i][j]] += 1
        
        except:
            continue

    for k, v  in tally.items():
        top500_counts[k].append(v)

top500_counts.items()


df_top500_words = pd.DataFrame(top500_counts)

full_data.join(df_top500_words, rsuffix= '_count' )

