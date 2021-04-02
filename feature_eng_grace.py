
import pandas as pd
import numpy as np 
import nltk 
import re 
from pprint import pprint
nltk.download('punkt')

#Mostly taking this stuff from Kayla's branch and playing around. Thanks Kayla!! 

df = pd.read_json('labeled_data.json')
df.info()
df.head()
df.text[0]

test = df.text[0]
print(len(test))
print(test[0:100])

#Text Normalization
test = test.lower()
test = re.sub(r'\d+', '', test) # remove numbers?? 
test = re.sub(r'[^\w\s]', ' ', test) # remove punctuation, make sure there is a space for slashes 
print(test)
# what else to normalize?

# Tokenization Use NLTK or Spacy ?? 
#Word tokenizing 

default_wt = nltk.word_tokenize
words = default_wt(test)
words[0:20]

#Stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
result = [i for i in words if not i in stop_words]
print(len(result))
result

#remove irrelevant words 
from calendar import month_abbr
mon_ab = list(month_abbr)

days=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
# need to put into lower case
months=['January','February','March', 'April','May','June','July','August','September','October','November','December']
# need to put into lower case

lowercase_days = {item.lower() for item in days}
lowercase_months = {item.lower() for item in months}
lowercase_mon_ab = {item.lower() for item in mon_ab}

exclusion_set = lowercase_days.union(lowercase_months).union(stop_words).union(lowercase_mon_ab)
# need to convert words to set for this to work
words = set(words)
cleaned = [w for w in words if w not in exclusion_set] #removing days and months
print(len(cleaned)) # From 2733 to 144 

#TRYING TO DO IT FOR EVERY TEXT 

#first make all necessary imports/ downloads
import nltk
nltk.download('punkt')
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')
from nltk.corpus import stopwords
from calendar import month_abbr

def clean(text):
    """ takes a block of text and cleans it by tokenizing it to words, making all lowercase, and removing punctuation, 
    stop words, numbers, days, and months. Returns a list of remaining words"""


    tokenizer = RegexpTokenizer(r'\w+')
    token_punc = tokenizer.tokenize(text)
    words = [word.lower() for word in token_punc]

    words = [word for word in words if word.isalpha()]

    mon_ab = list(month_abbr)

    days=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    months=['January','February','March', 'April','May','June','July','August','September','October','November','December']
    stop_words = set(stopwords.words('english'))
    lowercase_days = {item.lower() for item in days}
    lowercase_months = {item.lower() for item in months}
    lowercase_mon_ab = {item.lower() for item in mon_ab}

    exclusion_set = lowercase_days.union(lowercase_months).union(stop_words).union(lowercase_mon_ab)

    words = set(words)
    cleaned = [w for w in words if w not in exclusion_set] 
    return cleaned


df = pd.read_json('labeled_data.json')
df['words'] = [clean(text) for text in df.text]
df.text[0]
df.words[2]

# Lemmatization - getting to root words, rid of suffixes and affixes
# Makes more acurate matching??
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')
lemmatizer = WordNetLemmatizer()

for word in cleaned:
    lemmatizer.lemmatize(word)

lemmatizer.lemmatize("harvesting", pos = 'v')

#Have to do POS tagging to get it to do a good job bleh
# Taken from  https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

word = 'singing'
print(lemmatizer.lemmatize(word, get_wordnet_pos(word)))

print([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(cleaned)])

get_wordnet_pos(cleaned)

##Need to spend more time figuring out how to grab POS 