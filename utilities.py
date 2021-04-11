# This contains all the utility methods used to process the 
# Thomson Reuters News Data

from config import *
from glob import glob

import json
import pandas as pd
import numpy as np
import os

def get_all_filenames():
    folders = glob(os.path.join(DATA_PATH, '*'))
    years_files_dict = dict()
    for folder in folders:
        year = folder[-4:]
        years_files_dict[year] = glob(os.path.join(DATA_PATH, year, '*'))
    return years_files_dict


def load_large_json_file(filename):
    data = []
    fp = open(filename)
    lines = fp.readlines()
    fp.close()
    
    for line in lines[1:-1]:
        data.append(eval(line.strip()[:-1]))
    data.append(eval(lines[-1].strip()[:-2]))
    
    # Finally, package the list into a format the other methods will understand
    # (As a dictionary)
    data = {'Items': data}
    return data


def load_data(filename):
    fp = open(filename)
    try:
        data = json.load(fp)
    except:
        print(filename, 'too big. Handling line by line and not through conventional json loading.')
        data = load_large_json_file(filename)
    fp.close()
    return data


def filtered_article_count(data, language='en', subjects=None, subject_filter_type='any'):
    if not data:
        return None
    if subjects is None:
        return language_count(data, language)
    count = 0
    for art in data['Items']:
        if art['data']['language'] == language:
            if eval(subject_filter_type)(subject in art['data']['subjects'] for subject in subjects):
                count +=1
    return count


# PLACEHOLDERS: EVERYTHING COPIED FROM BIGDATA HW FROM LAST YEAR

# TODO: Check if we need all the imports!
# And is there another document term matrix api we can use? Not sure if we need all this (stemming probably not)

from sklearn.feature_extraction.text import CountVectorizer

# Use CountVectorizer's analyzer to process the articles
analyzer = CountVectorizer(stop_words='english').build_analyzer()

# The default analyzer doesn't stem, so we do some post-processing
def own_analyzer(docstr):
    return [stemmer.stem(wrd) for wrd in analyzer(docstr)]

# The countvectorizer needed while building Python's DTM 
cv = CountVectorizer(analyzer=own_analyzer)

            
def get_dtm_sparse_matrix(articles):
    dtm_raw = cv.fit_transform([art['data']['body'] for art in articles if art['data']['body'] != ''])
    return dtm_raw


def get_required_word_idx_freq_list(dtm_sparse):
    freq = dtm_sparse.sum(axis=0)  # sum along the columns --> returns a [1 x num cols] matrix
    # The order of this is the same as in the cv.vocabulary_ lookup table
    # Let's now associate the word label with its position in the dtm and its frequency count
    named_freq = [(wrd, idx, freq[0, idx]) for wrd, idx in cv.vocabulary_.items()]

    # Sort from highest to lowest frequency word (the order is preserved throughout)
    named_freq = sorted(named_freq, key = lambda xx: xx[2], reverse=True)
    # We only want the top 101-2100 frequent words 
    reqd = named_freq[101:2101]
    return reqd


def get_cosine_similarity_matrix(dtm_sparse, reqd_word_idx_list):    
    reqd_idx = [el[1] for el in reqd_word_idx_list]

    # get the m and c matrixes
    mm = dtm_sparse[:, reqd_idx]
    c1 = (mm.transpose()*mm).todense()  ## convert to dense matrix

    # calculate the cosine similarity matrix
    d1 = np.sqrt(np.diag(c1))
    d1 = np.reshape(d1, (len(d1), 1))  ## convert array to a Nx1 matrix
    d2 = d1.transpose() * d1
    cc = np.multiply(c1,1/d2)  ## this is element by element multiplication
    print('Shape of cosine similarity matrix:', cc.shape)
    return cc


def get_most_similar_words(cc, reqd_word_idx_freq_list, word_list, number=10):
    words_ordered = [wrd for wrd,_,_ in reqd_word_idx_freq_list] 
    for tgt_wrd in word_list:
        # Find index of the target word
        wrd_idx = words_ordered.index(tgt_wrd)                                      
        # Associate words with cosine similarity
        words_dict = {(wrd, cc[ii, wrd_idx]) for ii, wrd in enumerate(words_ordered)}  
        # Sort by cosine similarity measure
        words_dict = sorted(words_dict, key=lambda xx:xx[1],reverse=True)
        print(tgt_wrd, 'most similar:', words_dict[:number], end='\n\n')
