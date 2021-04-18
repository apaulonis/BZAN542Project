from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# Read the whole text.
import pandas as pd
text = pd.read_json('labeled_data.json', convert_dates= False)
x = ' '.join(text['text'].tolist())

# read the mask image
python_mask = np.array(Image.open(path.join(d, "pythonlogo_mask.png")))

stopwords = set(STOPWORDS)
stopwords = stopwords.union(set(STOPWORDS),{"said","pct","Reuters","Reuter"})

wc = WordCloud(background_color="white", max_words=2000, mask=python_mask,
               stopwords=stopwords, contour_width=3, contour_color='steelblue')

# generate word cloud
wc.generate(x)

# store to file
wc.to_file(path.join(d, "python_wc.png"))

# show
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
# plt.figure()
# plt.imshow(pythonlogo_mask, cmap=plt.cm.gray, interpolation='bilinear')
# plt.axis("off")
plt.show()