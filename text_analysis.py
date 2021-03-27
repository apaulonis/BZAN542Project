import nltk
from nltk.corpus import gutenberg

# Sample text to work with
alice = gutenberg.raw(fileids='carroll-alice.txt')
sample_text = '''We will discuss briefly about the basic syntax, structure and
design philosophies. There is a defined hierarchical syntax for Python code
which you should remember when writing code! Python is a really powerful
programming language!'''

len(sample_text)
len(alice)

# Sentence length of sample text
default_st = nltk.sent_tokenize
sample_sentences = default_st(text=sample_text)
alice_sentences = default_st(text=alice)
len(sample_sentences)
len(alice_sentences)

# Word tokenize
sentence = "The brown fox wasn't that quick and he couldn't win the race"
default_wt = nltk.word_tokenize
words = default_wt(sentence)
words

