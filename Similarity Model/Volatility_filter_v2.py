#!/apps/anaconda3/bin/python

import os
from config import *
import utilities as utils
import json
import importlib
import pandas as pd

def main():
    year_files_dict = utils.get_all_filenames()
    increment = os.environ['SGE_TASK_ID']
    year = str(START_YEAR - 1 + int(increment)) # Subtract 1 since the task ids start from 1
    
    # All Volatility words
    words = pd.read_csv(TEMP_PATH + '/volatility_words.csv')
    word_list = list(words[year])
    word_list.append('volatility')
    
    all_filenames = year_files_dict[year]
    all_volatility_articleids = dict()
    all_volatility_article_texts = dict()
    
    for fname in all_filenames:
        data = utils.load_data(fname)
        all_volatility_articleids[fname] = []
        all_volatility_article_texts[fname[-15:-9]] = []
        for art in data['Items']:
            # 1. Filter for english articles only
            if art['data']['language'] == 'en':
                # 2. Filter for articles about the US only
                if US_CODE in art['data']['subjects']:
                    # 3. Filter for articles about Volatility only - but with similar words now!
                    if any(word in art['data']['body'].lower() for word in word_list):
                        all_volatility_articleids[fname].append(art['data']['id'])
                        all_volatility_article_texts[fname[-15:-9]].append(art['data']['body'])
        print(fname[-15:-9], len(all_volatility_articleids[fname]))
        del data

    with open(TEMP_PATH + '/Volatility/Volatility_Articles_v2_%s.json' % year, 'w') as f:
        json.dump(all_volatility_articleids, f)

    with open(TEMP_PATH + '/Volatility/Volatility_Article_Texts_v2_%s.json' % year, 'w') as f:
        json.dump(all_volatility_article_texts, f)


if __name__ == '__main__':
    main()
