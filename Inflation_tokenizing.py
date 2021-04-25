#!/apps/anaconda3/bin/python

import os
from config import *
import utilities as utils
import json
import importlib
import pandas as pd

def main():
    increment = os.environ['SGE_TASK_ID']
    year = str(START_YEAR - 1 + int(increment)) # Subtract 1 since the task ids start from 1
    
    articles = utils.get_tokenized_articles(year, 'Inflation')
    print('Articles tokenized!')
    
    tok_articles_ev = utils.get_tokenized_articles_within_effective_vocab(articles)
    print('Tokenized articles with effective vocab done!')

    with open(TEMP_PATH + '/Inflation/Inflation_Articles_Tokenized_%s.json' % year, 'w') as f:
        json.dump(tok_articles_ev, f)

if __name__ == '__main__':
    main()
