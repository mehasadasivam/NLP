#!/apps/anaconda3/bin/python

from config import *

import gensim
import glob
import os
import re
import numpy as np
import pandas as pd
import time
import json

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim.models import ldaseqmodel


def main():

    THEME = 'Inflation'

    time_slices = pd.read_csv('Summary Stat Tables/%s_Article_Count.csv' % THEME, index_col=0)
    time_slices.sort_index(inplace=True)
    time_slices.index = pd.to_datetime(time_slices.index)
    # time_slices.groupby(time_slices.index.year)['No. of Volatility Articles'].sum()

    # Only upto 2018
    yearly_slices = time_slices.groupby(time_slices.index.year)['No. of %s Articles' % THEME].sum().values[:-2]

    # Load dictionary and corpus
    dictionary_all = gensim.corpora.Dictionary.load(TEMP_PATH + '/%s/%s_clean.dict' % (THEME, THEME))
    corpus_all = gensim.corpora.MmCorpus(TEMP_PATH + '/%s/%s_clean.mm' % (THEME, THEME))


    tic = time.time()

    ldaseq = ldaseqmodel.LdaSeqModel(corpus=corpus_all, 
                                     id2word=dictionary_all, 
                                     time_slice=yearly_slices, 
                                     passes=2,
                                     num_topics=10,
                                     em_min_iter=1,
                                     em_max_iter=1,
                                     chunksize=12000)
    
    print('LDA Seq Model Created. Time Taken: %d seconds' % int(time.time() - tic))
    
    # Save the model
    ldaseq.save(TEMP_PATH + '/%s/%s_LDASeqModel_yearly_10_Topics' % (THEME, THEME))
    
if __name__ == '__main__':
    main()
