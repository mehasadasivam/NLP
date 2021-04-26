#!/apps/anaconda3/bin/python

from config import *
import utilities as utils
import json
import pandas as pd
import gensim
import time

from gensim.models import LdaSeqModel


def main():

    THEME = 'Volatility'
    tic = time.time()

#     # 1. Combine all the list of lists of tokenized articles within effective 
#     # vocabulary each year 

#     years = [str(year) for year in range(START_YEAR, END_YEAR + 1)]

#     all_tok_articles = []
#     for year in years:
#         with open(TOKENIZED_ARTICLES_PATH % (THEME, THEME, year)) as f:
#             all_tok_articles.extend(json.load(f))
#     print(TOKENIZED_ARTICLES_PATH % (THEME, THEME, year), 'done!')
    
#     # 2. Create the corpus and dictionary - This takes a while
#     dictionary_all = gensim.corpora.Dictionary(all_tok_articles)
#     corpus_all = [dictionary_all.doc2bow(doc) for doc in all_tok_articles]

#     # 3. Store the corpus and dictionary in case of issues in the processing beyond this
#     dictionary_all.save(TEMP_PATH + '/%s/%s.dict' % (THEME, THEME))
#     gensim.corpora.MmCorpus.serialize(TEMP_PATH + '/%s/%s.mm' % (THEME, THEME), corpus_all)

    # If the above are already complete - load the corpus and dictionary
    dictionary_all = gensim.corpora.Dictionary.load(TEMP_PATH + '/%s/%s.dict' % (THEME, THEME))
    corpus_all = gensim.corpora.MmCorpus(TEMP_PATH + '/%s/%s.mm' % (THEME, THEME))
    print('Corpus and Dictionary are in local memory now. Time Taken:', time.time() - tic)
    
    # 4. Get the time slices (The article count per month)
    article_count = pd.read_csv('Summary Stat Tables/%s_Article_Count.csv' % THEME, index_col=0)
    time_slices = article_count.sort_index()['No. of %s Articles' % THEME].values
    
#     # The above is erroring out due to memory issues (300 slices seems to be too much for it?)
#     # Trying yearly
#     article_count.index = pd.to_datetime(article_count.index)
#     time_slices = article_count.groupby(article_count.index.year)['No. of %s Articles' % THEME].sum().values
 
    
    tic = time.time()
    # 5. Create the SeqLDA Model
    ldaseq = LdaSeqModel(corpus=corpus_all, id2word=dictionary_all, time_slice=time_slices, num_topics=10)   
    
    print('LDA Seq Model Created. Time Taken: %d seconds' % int(time.time() - tic))
    
    # 6. Save the model
    ldaseq.save(TEMP_PATH + '/%s/%s_LDASeqModel_monthly' % (THEME, THEME))
    
if __name__ == '__main__':
    main()
