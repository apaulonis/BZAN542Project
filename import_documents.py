'''
Use BeautifulSoup to parse the standardized general markup (.sgm) files of 
text from Reuters news articles.  Each news "document" has associated 
metadata in markup tags that needs to be parsed into a data structure.  Parse
the raw data into a dataframe and then export it as a json file.
'''
import os
from bs4 import BeautifulSoup, UnicodeDammit #pip install beautifulsoup4
# Note that lxml needs to be installed, but not imported
import pandas as pd
import numpy as np
import re

# Main class to do all the work
class Reuters():
    def __init__(self):
        # Initialize a dataframe with columns corresponding to the main markup
        # tags in the raw data files.
        self.data = pd.DataFrame( {'newid': [np.nan], 
                            'date': [np.nan], 
                            'topics': [np.nan], 
                            'places': [np.nan], 
                            'people': [np.nan], 
                            'orgs': [np.nan], 
                            'exchanges': [np.nan], 
                            'companies': [np.nan], 
                            'title': [np.nan]} )
        self.soup = None
        self.paths = None

    # Create a list of the filepaths to the reut2-xxx.sgm files containing the
    # raw data.  These files are in a raw_data folder within the project.
    def make_paths(self):
        self.paths = []
        for ele in os.listdir('raw_data'):
            if ele.endswith('.sgm'):
                self.paths.append('raw_data/' + ele)

    # For a given raw data file, read it in and then create a BeautifulSoup
    # object from the contents.  Ensure that the contents of the file are
    # converted to unicode.  Use the fast lxml parser rather than the 
    # Beautiful Soup default.
    def make_soup(self, path):
        fheader = open(path, 'r')
        filetext = fheader.read()
        fheader.close()
        self.soup = BeautifulSoup(UnicodeDammit(filetext).unicode_markup, features= "lxml")
        
    # Parse the raw data file and for each record, put the various attributes
    # into a dataframe
    def get_documents(self):

        df_import = self.data
        
        # For a specific indexed record (inside the <REUTERS></REUTERS> tags) in
        # the BeautifulSoup object, create a dictionary with all the child tags
        # and their content.
        def get_children(text, document_index, data_dictionary = {}):
            tag_text = data_dictionary
            current_tag = None
            current_text = None
            # Get all child tags and their content from the full Reuters document
            # at the index position.  Loop through all children.
            for child in text.find_all('reuters')[document_index].findChildren():
                # Ignore children with certain tag names
                if child.name in ['unknown', 'd', 'mknote']:
                    continue
                # Extract the tag name and the contents
                current_tag = child.name
                # If in a tag that can have multiple values, create a list of
                # the contents with markup removed
                if current_tag in ['topics', 'places', 'people', 'orgs', 'exchanges', 'companies']:
                  current_contents = [item for item in child.stripped_strings]
                # If in the text tag, return contents with starting or ending
                # whitespace and newlines removed
                elif current_tag == 'text':
                  current_contents = re.sub(r'\s+$', '', child.text)
                  current_contents = re.sub(r'^\s+', '', current_contents)
                  current_contents = re.sub(r'\n', ' ', current_contents)
                # Otherwise, just return the contents
                else:
                  current_contents = child.text
                
                # Add the tag and contents to the dictionary
                tag_text[current_tag] = current_contents
            return tag_text
            
        # Loop through each top-level document in the data file as delimited
        # by the <REUTERS></REUTERS> tags.
        for i, doc in enumerate(self.soup.find_all('reuters')):
            # Show progress in the terminal
            print('\r' + str(i), end='')
            # Start the dictionary for the document with the NEWID attribute
            tag_text = {'newid': str(doc.get("newid") ) }
            # Build the rest of the dictionary for all the other tags and content
            # using the get_children function
            tag_text = get_children(self.soup, i, data_dictionary= tag_text)
            # Convert the dictionary into a new row in the master dataframe
            df_import = df_import.append(tag_text, ignore_index= True )

        self.data = df_import

if __name__ == '__main__':
    # Initalize a new docs object
    docs = Reuters()
    # Create a list of all the raw data files in the raw_data directory
    docs.make_paths()
    # For each file, convert the markup text into a BeautifulSoup object and
    # then parse all of the attributes of each document within the object into
    # dataframe rows
    for filepath in docs.paths:
        print(filepath)
        docs.make_soup(filepath)
        docs.get_documents()
        print(docs.data.tail())

    # Drop the 0th row of the dataframe with blanks
    docs.data.drop(0, axis= 0,inplace=True)
    # Reset the index
    docs.data.reset_index(drop=True, inplace=True)

    # Convert the dataframe to json and write it to a file
    docs.data.to_json('clean_data.json')

