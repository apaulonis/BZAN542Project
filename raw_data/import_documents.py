import os
from sys import path
from bs4 import BeautifulSoup, UnicodeDammit #pip install beautifulsoup4
import lxml  #pip install lxml, used within beautiful soup call 'features'
import pandas as pd
import numpy as np

class Reuters():
    def __init__(self):
        self.data = pd.DataFrame( {'newid': [np.nan], 
                            'date': [np.nan], 
                            'topics': [np.nan], 
                            'places': [np.nan], 
                            'people': [np.nan], 
                            'orgs': [np.nan], 
                            'exchanges': [np.nan], 
                            'companies': [np.nan], 
                            'unknown': [np.nan], 
                            'title': [np.nan]} )
        self.soup = None
        self.paths = None

    def make_paths(self):
        self.paths = []
        for ele in os.listdir('raw_data'):
            if ele.endswith('.sgm'):
                self.paths.append('raw_data/' + ele)

    def make_soup(self, path):
        fheader = open(path, 'r')
        filetext = fheader.read()
        fheader.close()
        self.soup = BeautifulSoup(UnicodeDammit(filetext).unicode_markup, features= "lxml" )
        

    def get_documents(self):

        df_import = self.data
        
        def get_children(text, document_index, data_dictionary = {}):
            tag_text = data_dictionary
            current_tag = None
            current_text = None
            for child in text.find_all('reuters')[document_index].findChildren():
                if child.name in ['d', 'mknote']:
                    continue
                current_tag = child.name
                current_text = child.text

                if current_tag not in tag_text:
                    tag_text[current_tag] = []
                
                tag_text[current_tag].append(current_text)
            return tag_text
            
        for i, doc in enumerate(self.soup.find_all('reuters')):
            tag_text = {'newid': str(doc.get("newid") ) }
            tag_text = get_children(self.soup, i, data_dictionary= tag_text)    
            df_import = df_import.append(tag_text, ignore_index= True )

        self.data = df_import

if __name__ == '__main__':
    docs = Reuters()
    docs.make_paths()
    for filepath in docs.paths:
        print(filepath)
        docs.make_soup(filepath)
        docs.get_documents()
        print(docs.data.tail())

    docs.data.drop(0, axis= 0,inplace=True)

    docs.data.to_json('dirty_data.json')

