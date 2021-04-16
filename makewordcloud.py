from wordprocessor import WordProcessor
import pandas as pd

from wordcloud import WordCloud
import matplotlib.pyplot as plt


full_data = pd.read_json('labeled_data.json', convert_dates= False)
total_corpus = WordProcessor(full_data)

total_corpus.get_clean_words()

def plot_cloud(wordcloud, size = (12,8)):
    plt.figure(figsize = size)
    plt.imshow(wordcloud)
    plt.axis('off')


def cloud_structure(topic):
    topic_indices= list(total_corpus.data.loc[total_corpus.data['topic_label'] == topic].index)
    topic_words = []

    for ind in topic_indices:
        doc = [ word for word in total_corpus.clean_stemmed[ind] ]
        topic_words = list(topic_words) + list(doc)

    topic_words = ' '.join(list(topic_words))
    return topic_words

for topic in full_data['topic_label'].unique():
    print(topic)
    topic_words = cloud_structure(topic)

    wc = WordCloud(width = 1200, height=800, background_color = 'white', colormap = 'Set2' ).generate_from_text(topic_words)
    plot_cloud(wc)
    plt.savefig(topic + '.png', dpi = 600)



