from numpy.core.numeric import full_like
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer

full_data = pd.read_json('labeled_data.json')

full_data.info()

#Stop words from nltk, punctuation from string
stop_words = set(stopwords.words('english'))
punc = {i for i in string.punctuation}

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

        


        




word_tokens = nltk.tokenize.word_tokenize(self.all_words)

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
    clean_words.append([word.lower() for word in all_words if word not in stop_words and word not in punc])

len(clean_words)


## No words common to EVERY article
set_clean_words = set(clean_words[0])
for s in clean_words[1:]:
    set_clean_words.intersection_update(s)
list(set_clean_words)


# Adding a column of the clean words, tokenized by word
full_data['clean_text'] = clean_words

full_data['clean_text'].head()

len(clean_words)


for i in range(len(full_data['clean_text'])):
    clean_tagged = nltk.pos_tag(full_data['clean_text'][i])
    print(clean_tagged)

pos_dict = dict()
for i in range(len(clean_tagged)):
    if clean_tagged[i][1] not in pos_dict.keys():
        pos_dict[clean_tagged[i][1]] = 1
        continue

    pos_dict[clean_tagged[i][1]] += 1

pos_proportions = dict()
for k, v in pos_dict.items():
    pos_proportions[k] = v/len(clean_words)

