# Text classification example (NLP) - BZAN 542

This repo contains an example of using Python for text classification, a common natural language processing (NLP) problem. The example was built for a group project in BZAN 542 - Data Mining Methods for Business Applications at the University of Tennessee in Spring 2021. It is being posted here as a reference and hopefully a springboard for future extensions and refinements.

## Contacts

- [Andrew Paulonis](mailto:apaulonis@gmail.com)
- James McHale
-
- 
-
-

## Overview

This example uses the [Reuters-21578 data set](https://archive.ics.uci.edu/ml/datasets/Reuters-21578+Text+Categorization+Collection). This is a pre-labeled set of 21578 short news documents from Reuters, collected in 1987. Documents may have multiple labels. Not all documents had a label, so only the labeled documents in the corpus were used for modeling.

The machine learning problem is to predict a document label from the document text. There is additional metadata associated with many of the documents in the corpus, but none of that was used for modeling in this study. Because the documents may have multiple labels, it would be possible to create a model which predicts multiple labels for a given document. However, for this class project, a simpler multi-class classification problem was solved. A pre-processing step was conducted to aggregate the raw, possibly multiple labels into a single "subtopic" label per document. Model training and prediction was then conducted using the subtopic labels.

Four methods for text classification were studied:

- Feature generation using TF-IDF and modeling using SVC (support vector classification)
- Feature generation using spaCy embedded word vectors and modeling using SVC
- Feature generation using { several things } and modeling using random forest
- Feature generation using { several things } and modeling using K-nearest neighbors (KNN)

Excellent results in predicting label values for the held-out test documents were obtained:

| Method | Test Fraction | F1-micro on Test |
| -- | -- | -- |
| TF-IDF uni-grams / SVC | 0.25 | 0.944 |
| spaCy word vectors / SVC | 0.25 | 0.923 |
| TF-IDF+ / random forest | 0.20 | ??? |
| TF-IDF+ / KNN | 0.20 | ??? |

## Data Preprocessing

The Reuters data is provided as a set of 22 standardized general markup (.sgm) files. Each file contains 1000 documents except the 22nd, which contains 578. Together there are 21,578 documents across all files.

The general markup in these files is similar to HTML or XML markup and it can be parsed using methods that are typical for those types of files. We used BeautifulSoup (bs4) to do the parsing.

The .sgm files were extracted from the Reuters data archive and placed in the `raw_data` folder. The Python file `import_documents.py` is used to read the .sgm files within that folder, parse the SGM contents into a structured table form, and then export a JSON file `clean_data.json` with the structured data. As part of this import, the document text had newline characters replaced with spaces and whitespace was trimmed from the start and end of the text.

## Subtopic Relabeling

As this class project was the first foray into NLP for everyone in our group, we decided not to tackle the multi-label classification problem, but rather the simpler multi-class classification problem. The raw Reuters data can have multiple labels per document. In order to aggregate to a single label per document, we used information from the `cat-descriptions_120396.txt` file included with the Reuters archive (and extracted in the `raw_data` folder). This file shows how some of the topic labels can be aggregated to higher-level categories.

The Python file `feature_eng.py` recodes document labels into six categories

- money
- econ
- acquisitions
- earnings
- commodity
- energy

For the vast majority of documents, the raw labels mapped to only one of the aggregated labels so the recoding was straightforward. In a few cases, raw labels mapped to more than one aggregated label. The recoded label was selected as the one with higher count of associated raw labels. In the very rare event that there was a tie in the highest count of raw labels mapping to a recoded label, the recoded label was selected randomly from those with the tied high count.

## Support Vector Classification

The Python file `SVCmodels.py` implements the two methods of modeling with support vector classification described in the overview.

In both methods, the scikit-learn [OneVsRestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html?highlight=onevsrest#sklearn.multiclass.OneVsRestClassifier) function was used to handle the multi-class classification problem. The scikit-learn [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) algorithm was used to perform classification. Support vector machines are known to work well for text classification problems \[[Joachims](https://www.cs.cornell.edu/people/tj/publications/joachims_97b.pdf)\].

For both methods, the set of labeled documents from the Reuters set was randomly split into a training set of 75% of the documents and a test set of 25%.

Initial model fitting was done with default parameters for LinearSVC. After verifying that these default parameters produced excellent results, some parameter tuning was attempted using scikit-learn [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html?highlight=randomized%20search%20cv#sklearn.model_selection.RandomizedSearchCV). There are not many parameters to optimize over in LinearSVC, and some combinations of parameters are not compatible. Optimization was still attempted, however. Results from optimization were nearly identical to the use of default parameters for this algorithm and this use case.

### TF-IDF with SVC

A very popular choice of feature generation for text modeling is [TF-IDF](http://tfidf.com/) (term frequency - inverse document frequency). The scikit-learn function [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) was used to perform TF-IDF.

All default parameters of TfidfVectorizer were used except for tokenization. Since the text analytic package spaCy will be loaded for a different kind of feature generation described below, we chose to use the spaCy lemmatizer and stopword set for tokenization. The expectation was that these would be as good or better than anything available in scikit-learn. Tokenization steps included:

- convert all words to lower case
- do not include words less than 3 characters
- filter out default spaCy stopword set plus the word "reuter"
- replace all words with their spaCy lemma
- remove words with numbers

The TfidfVectorizer default is to generate only uni-grams (single-word tokens). Including bi-grams and tri-grams was considered, but the model performance including only uni-grams was very acceptable and the increased computing resources needed for a larger TF-IDF matrix were not justified.

TfidfVectorizer outputs a sparse TF-IDF matrix which is accepted as input to LinearSVC. This is important since even with uni-grams only, the TF-IDF training matrix was well in excess of 8,000 rows and 20,000 columns. That matrix is mostly zeros since a single document (row of TF-IDF) has only a tiny fraction of all tokens (TF-IDF columns) from the entire corpus of documents. Making use of sparse matrices dramatically reduces memory requirements and computation time.

TF-IDF with SVC did a great job on the Reuters text classification, generating an F1 micro-averaged score of over 0.94 on the test set.

### spaCy word vectors with SVC

Another way to generate features from a text document for use in modeling is to use word vectors. The Python package spaCy includes [pre-trained language models where words are encoded as 300-dimensional numeric vectors](https://spacy.io/usage/linguistic-features#vectors-similarity) that represent the relationship of that word to an entire vocabulary. Documents can be represented as vectors as well by averaging the vectors of all the words in the document.

The spaCy pipeline [en_core_web_md](https://spacy.io/models/en#en_core_web_md) was used in conjunction with document tokenization to produce token vectors and averaged document vectors for the Reuters documents. Tokens included single words of 3 or more characters, not containing numbers, and not in the spaCy stopword list (including "reuter").

The resulting feature matrix is 300 columns wide because that is the dimension of the spaCy language model. This feature matrix is not sparse since all or almost all of the vector elements for every token are non-zero. However, working with a 8,000 x 300 full matrix is much less challenging than working with a full matrix of 8,000 x 20,000, which would be the case if the TF-IDF matrix was not sparse.

Word/document vectors with SVC did very well on the Reuters text classification, generating an F1 micro-averaged score of greater than 0.92. TF-IDF did a little better. One possible reason is that in this document corpus, the document label is often included in the document text and the text often does not include other labels. For example, earnings reports often include the word earnings. Documents related to energy may often include the word energy. However earnings reports do not often include the word energy. TF-IDF provides a fairly direct way to provide this kind of relationship to the model. Because word vectors are averaged together into document vectors, the word vector approach represents the relationship in a much more indirect way.

## Random Forest Classification

James put NLTK / random forest description here

## K-Nearest Neighbors Classification

James put NLTK / KNN description here

## Python Environment Requirements

The code is written in Python 3. Required packages as well as their dependencies include:

- bs4
- lxml
- matplotlib
- nltk
- numpy
- pandas
- plotly
- scikit-learn
- scipy
- spacy

Install these packages with pip or conda, depending on the flavor of Python you have installed.

After installing NLTK, it is necessary to install the NLTK model files as well. If you are not concerned about disk storage, you can install all NLTK data with the following commands from a Python prompt:
```
> import nltk
> nltk.download('all')
```
If you would like to be more conservative, you can use `nltk.download()` to pop up an NLTK Downloader window. From the window, select the Models tab. Choose a model file and click the Download button. Repeat for all model files in the tab and then close the Downloader window.

After installing spaCy, it is necessary to install the language model pipeline. From an operating system command prompt:
```
$ python -m spacy download en_core_web_md
```

## Running the Programs

There are four steps to produce the project results:

1. Run `import_documents.py` in the main folder - this will produce a data file `clean_data.json`. If the file already exists, it will be overwritten. The Reuters .sgm files must be present in a subfolder `raw_data`.
2. Run `feature_eng.py` in the main folder - this requires `clean_data.json` as input. It produces `labeled_data.json` as output. If the file already exists, it will be overwritten.
3. Run `SVCmodels.py` - this requires `labeled_data.json` as input. It will produce text and graph output in the terminal for the two support vector modeling approaches.
4. Run `randomforest_knn_modeling.py` - this requires `labeled_data.json` as input. It will produce text and graph output in the terminal for the random forest and KNN modeling approaches and produce output files `candidate_models.json`, `candidate_models_2.json`, `knn_tune.json`, `knn_tune_2.json`, and `knn_tuned.json`.