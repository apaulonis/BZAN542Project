import pandas as pd

df = pd.read_json('labeled_data.json')
df.info()


#playing with tokenization - sentences 

import nltk
nltk.download('punkt')

testtext = df.text[0]

token_Sent = nltk.sent_tokenize
sentences = token_Sent(testtext)
sentences[0]

##bigrams?



#playing with tokenization - words

token_Word = nltk.word_tokenize
words = token_Word(testtext)
words[0:10]

##removing punctuation

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
token_punc = tokenizer.tokenize(testtext)
words = [word.lower() for word in token_punc] #making all lower case

#removing stop words - included in next step

nltk.download('stopwords')
from nltk.corpus import stopwords

##removing numbers and months (not sure if we want #s and months removed, but wanted to try)

words = [word for word in words if word.isalpha()] #removing numbers

from calendar import month_abbr
mon_ab = list(month_abbr)

days=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
# need to put into lower case
months=['January','February','March', 'April','May','June','July','August','September','October','November','December']
# need to put into lower case

stop_words = set(stopwords.words('english'))
lowercase_days = {item.lower() for item in days}
lowercase_months = {item.lower() for item in months}
lowercase_mon_ab = {item.lower() for item in mon_ab}

exclusion_set = lowercase_days.union(lowercase_months).union(stop_words).union(lowercase_mon_ab)

# need to convert words to set for this to work
words = set(words)
cleaned = [w for w in words if w not in exclusion_set] #removing days and months
cleaned #list of words with no punctuation, stop words, months, days, or numbers



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

df['words'] = [clean(text) for text in df.text]
df.info()

df.words[20]


