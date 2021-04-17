from wordprocessor import WordProcessor
import pandas as pd
import numpy as np
from collections import Counter
#from PIL import Image #The python logo turned into a circle.  I tried using a book, then some other stuff.  It will take more time messing with to make it look good than I'm willing to put in
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#Required set up
full_data = pd.read_json('labeled_data.json', convert_dates= False)
total_corpus = WordProcessor(full_data, train_size= 1)
total_corpus.get_clean_words()
total_corpus.get_word_sets()

def wc_tf_idf(topic):
    """
    Counts words when passed a topic labe.
    """
    topic_indices = list(full_data.loc[full_data['topic_label'] == topic].index)
    all_words = dict()
    all_docs = dict()

    for word in total_corpus.word_set:
        all_words[word] = 0
        all_docs[word] = 0
        #word_counter_master[word] = 0

    topic_clean_stemmed = [total_corpus.clean_stemmed[i] for i in topic_indices]

    for i in range(len(topic_clean_stemmed)):
        #word_count = word_counter_master.copy()
        word_count = Counter()

        for word in topic_clean_stemmed[i]:
            try:
                word_count[word] += 1
            
            except:
                pass

        for k, v in word_count.items():
            try:
                all_words[k] += (v) #/ len(topic_clean_stemmed[i]))
                all_docs[k] += ((v > 0) * 1)

            except:
                pass


    tf_idf = dict()
    for k, v in all_words.items():
        tf_idf[k] = v * np.log( len( topic_clean_stemmed ) / (1+all_docs[k]) )

    return tf_idf

def plot_cloud(wordcloud, size = (12,8)):
    """
    generate word cloud
    """
    plt.figure(figsize = size)
    plt.imshow(wordcloud)
    plt.axis('off')

def cloud_structure(topic):
    """
    Setup for generating Word Cloud
    """
    topic_indices= list(total_corpus.data.loc[total_corpus.data['topic_label'] == topic].index)
    topic_words = []

    for ind in topic_indices:
        doc = [ word for word in total_corpus.clean_stemmed[ind] ]
        topic_words = list(topic_words) + list(doc)

    topic_words = ' '.join(list(topic_words))
    return topic_words

if __name__ == '__main__':
    for topic in full_data['topic_label'].unique():
        print(topic)
        topic_words = cloud_structure(topic)

        wc = WordCloud(width = 1200, height=800, background_color = 'linen', colormap = 'Dark2' ).generate_from_text(topic_words)
        plot_cloud(wc)
        plt.savefig('wordcloud_' + topic + '.png', dpi = 600)

        tf_idf = wc_tf_idf(topic)
        wc = WordCloud(width = 1200, height=800, background_color = 'linen', colormap = 'Dark2' ).generate_from_frequencies(tf_idf)
        plot_cloud(wc)
        plt.savefig('wordcloud_' + topic + 'tfidf.png', dpi = 600)

