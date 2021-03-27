import nltk
from nltk.corpus import gutenberg
alice = gutenberg.raw(fileids='carroll-alice.txt')
sample_text = '''We will discuss briefly about the basic syntax, structure and
design philosophies. There is a defined hierarchical syntax for Python code
which you should remember when writing code! Python is a really powerful
programming language!'''

len(sample_text)
len(alice)