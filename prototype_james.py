from collections import Counter
import string
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
import numpy as np



class WordProcessor:
    def __init__(self, df):
        super().__init__()

        self.data = df[['topic_label', 'text', 'date']]
        self.train_rows = np.random.choice(self.data.index, size = int(.8*len(self.data.index)), replace= False)
        self.train_rows.sort()

    def get_clean_words(self, verbose = False):
        stop_words = set(stopwords.words('english'))
        punc = {i for i in string.punctuation}
        punc = punc.union({'--', '``', "''"})
        porter_stem = nltk.PorterStemmer()

        self.clean_words = []
        self.clean_stemmed = []
        self.clean_stemmed_no_nums = []

        for i in self.data.index:
            all_words = nltk.tokenize.word_tokenize(self.data['text'][i])
            
            
            self.clean_words.append( [word.lower() for word in all_words if word.lower() not in stop_words and word.lower() not in punc])

            self.clean_stemmed.append([porter_stem.stem(word.lower()) for word in 
                nltk.tokenize.word_tokenize(re.sub(r'\s+', ' ', re.sub(r'\d+', '', self.data['text'][i]))) if word.lower() not in stop_words and word.lower() not in punc ])


            if verbose:
                if i % 2835 == 0:
                    print('iteration no: ', i , 'of ', len(self.data.index))

    def get_parts_of_speech(self, verbose = False):
        """
        Generate features for part of speech proportions
        
        Columns appended to self.data
        """

        #Stop words, punctuation definitions
        pos_comprehensive = ['CC','CD', 'DT','EX', 'FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP','RB','RBR','RBS','RP','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB', 'EXCEPT']
        pos_counter = {key: 0 for key in pos_comprehensive}
        #all_counts = []
        all_proportions = []
        
        for i in self.data['text'].index:
            pos_dict = pos_counter.copy()

            clean_tagged = nltk.pos_tag(self.clean_words[i])

            for j in range(len(clean_tagged)):

                try:
                    pos_dict[clean_tagged[j][1]] += 1

                except:
                    pos_dict['EXCEPT'] += 1

            pos_proportions = dict()
            for k, v in pos_dict.items():
                pos_proportions[k] = v/len(clean_tagged)
            
            #all_counts.append(pos_dict)
            all_proportions.append(pos_proportions)

            if verbose:
                if i % 2835 == 0:
                    print('iteration no: ',i, 'of ', len(self.data['text'].index))
        
        df_pos_prop = pd.DataFrame(all_proportions)
        self.data = self.data.join(df_pos_prop, rsuffix= '_prob' )

        #self.all_counts = all_counts
        #self.all_proportions = all_proportions

    def get_NE_gen(self, verbose = False):
        self.entities = set()
        all_entities = []

        for i in self.data.index:
            sentences = nltk.tokenize.sent_tokenize(self.data['text'][i])
            sentences = [nltk.word_tokenize(sent) for sent in sentences]
            sentences = [nltk.pos_tag(sent) for sent in sentences]

            list_tracker = []

            for sent in sentences:
                NE_chunk = nltk.ne_chunk(sent, binary = True)

                for el in NE_chunk.subtrees():
                    if el.label() == 'NE':
                        list_tracker.append(' '.join([w.lower() for w,t in el]))

                        if i in self.train_rows:
                            self.entities.update(list_tracker)
            
            all_entities.append(list_tracker)

            if verbose:
                if i % 1000 ==0:
                    print(i, 'processsed of', len(self.data.index))


        NE_in_doc = dict()
        NE_count = dict()

        for el in self.entities:
            NE_in_doc[el] = np.zeros(len(self.data.index), dtype= np.int8)
            NE_count[el] = 0


        for j in range(len(all_entities)):
            NE_tally = NE_count.copy()

            for el in all_entities[j]:
                try:
                    NE_tally[el] += 1
                except:
                    pass

            for k, v in NE_tally.items():
                try:
                    NE_in_doc[k][j] = v

                except:
                    pass
                
            if verbose:
                if j % 1000 ==0:
                    print('iteration no: ',j, 'of ', len(self.data['text'].index), ' final pass')
        
        self.data = self.data.join(pd.DataFrame(NE_in_doc), rsuffix= '_entity' )

    def get_word_sets(self):
        if not hasattr(self, 'clean_stemmed_no_nums'):
            self.get_clean_words()
        
        self.word_set = set()
        commodity_set = set()
        earnings_set = set()
        acquisitions_set = set()
        econ_set = set()
        money_set = set()
        energy_set = set()

        for i in self.train_rows:

            working_label = self.data['topic_label'][i]
            
            self.word_set.update(self.clean_stemmed[i])

            if working_label == 'commodity':
                commodity_set.update(self.clean_stemmed[i])

            if working_label == 'earnings':
                earnings_set.update(self.clean_stemmed[i])

            if working_label == 'acquisitions':
                acquisitions_set.update(self.clean_stemmed[i])

            if working_label == 'econ':
                econ_set.update(self.clean_stemmed[i])

            if working_label == 'money':
                money_set.update(self.clean_stemmed[i])

            if working_label == 'energy':
                energy_set.update(self.clean_stemmed[i])

        self.unique_commodity = commodity_set-(earnings_set|acquisitions_set|econ_set|money_set|energy_set)
        self.unique_earnings = earnings_set-(commodity_set|acquisitions_set|econ_set|money_set|energy_set)
        self.unique_acquisitions = acquisitions_set-(earnings_set|commodity_set|econ_set|money_set|energy_set)
        self.unique_econ = econ_set-(acquisitions_set|earnings_set|commodity_set|money_set|energy_set)
        self.unique_money = money_set-(econ_set|acquisitions_set|earnings_set|commodity_set|energy_set)
        self.unique_energy = energy_set-(econ_set|acquisitions_set|earnings_set|commodity_set|money_set)

    def get_counts(self, verbose = False):
        if not hasattr(self, 'word_set'):
            self.get_word_sets()
        
        self.all_words = dict()
        self.all_docs = dict()
        word_counter_master = dict()

        category_tracker = {
            'commodity' : [],
            'earnings' : [],
            'acquisitions' : [],
            'econ' : [],
            'money' : [],
            'energy' : [],
            'commodity_dummy' : [],
            'earnings_dummy' : [],
            'acquisitions_dummy' : [],
            'econ_dummy' : [],
            'money_dummy' : [],
            'energy_dummy' : []
        }

        category_tracker['acquisitions'] = np.zeros(len(self.clean_stemmed), dtype= np.float32)
        category_tracker['commodity'] = np.zeros(len(self.clean_stemmed), dtype= np.float32)
        category_tracker['earnings'] = np.zeros(len(self.clean_stemmed), dtype= np.float32)
        category_tracker['econ'] = np.zeros(len(self.clean_stemmed), dtype= np.float32)
        category_tracker['energy'] = np.zeros(len(self.clean_stemmed), dtype= np.float32)
        category_tracker['money'] = np.zeros(len(self.clean_stemmed), dtype= np.float32)

        category_tracker['acquisitions_dummy'] = np.zeros(len(self.clean_stemmed), dtype= np.int8)
        category_tracker['commodity_dummy'] = np.zeros(len(self.clean_stemmed), dtype= np.int8)
        category_tracker['earnings_dummy'] = np.zeros(len(self.clean_stemmed), dtype= np.int8)
        category_tracker['econ_dummy'] = np.zeros(len(self.clean_stemmed), dtype= np.int8)
        category_tracker['energy_dummy'] = np.zeros(len(self.clean_stemmed), dtype= np.int8)
        category_tracker['money_dummy'] = np.zeros(len(self.clean_stemmed), dtype= np.int8)

        for word in self.word_set:
            self.all_words[word] = np.zeros(len(self.clean_stemmed), dtype= np.float32)
            self.all_docs[word] = 0
            #word_counter_master[word] = 0

        for i in range(len(self.clean_stemmed)):
            #word_count = word_counter_master.copy()
            word_count = Counter()

            if verbose:
                if i % 100 == 0:
                    print('iteration ', i, ' of ', len(self.clean_stemmed))

            for word in self.clean_stemmed[i]:
                try:
                    word_count[word] += 1
                
                except:
                    pass

            for k, v in word_count.items():
                try:
                    self.all_words[k][i] = (v / len(self.clean_stemmed[i]))
                    self.all_docs[k] += ((v > 0) * 1)

                    category_tracker['acquisitions'][i] += v * (k in self.unique_acquisitions)
                    category_tracker['commodity'][i] += v * (k in self.unique_commodity)
                    category_tracker['earnings'][i] += v * (k in self.unique_earnings)
                    category_tracker['econ'][i] += v * (k in self.unique_econ)
                    category_tracker['energy'][i] += v * (k in self.unique_energy)
                    category_tracker['money'][i] += v * (k in self.unique_money)
                except:
                    pass

        category_tracker['acquisitions_dummy'] =  (category_tracker['acquisitions'] > 0) * 1 
        category_tracker['commodity_dummy'] = (category_tracker['commodity'] > 0) * 1 
        category_tracker['earnings_dummy'] = (category_tracker['earnings'] > 0) * 1 
        category_tracker['econ_dummy'] = (category_tracker['econ'] > 0) * 1 
        category_tracker['energy_dummy'] = (category_tracker['energy'] > 0) * 1 
        category_tracker['money_dummy'] = (category_tracker['money'] > 0) * 1 

        category_tracker = pd.DataFrame(category_tracker)

        self.data = self.data.join(category_tracker, rsuffix = '_words')

    def get_tf_idf(self):
        """
        Term adjusted frequency * log(Ndocs total/ Ndocs with term)
        
        (count term) / (# words in doc)
        """
        tf_idf = dict()
        for k, v in self.all_words.items():
            tf_idf[k] = v * np.log( len( v ) / self.all_docs[k] )

        if hasattr(self, 'all_words'):        
            delattr(self, 'all_words')
            delattr(self, 'all_docs')

        tf_idf = pd.DataFrame(tf_idf)

        self.data = self.data.join(tf_idf, rsuffix = '_tfidf')

    def get_all(self):
        self.get_clean_words()
        self.get_parts_of_speech()
        self.get_word_sets()
        self.get_counts()
        self.get_tf_idf()
        self.get_NE_gen()

if __name__ == '__main__':
    full_data = pd.read_json('labeled_data.json', convert_dates= False)

    #Cleaning Date
    full_data['date'] = full_data['date'].str.slice(start = 0, stop = 20)
    full_data['date'] = pd.to_datetime(full_data['date'])

    obj = WordProcessor(full_data)

    obj.get_all()

    obj.data.drop('text',axis = 1)
    print(obj.data.shape)
