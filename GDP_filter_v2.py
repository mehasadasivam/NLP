#!/apps/anaconda3/bin/python

import os
from config import *
import utilities as utils
import json
import importlib


def main():
    year_files_dict = utils.get_all_filenames()
    increment = os.environ['SGE_TASK_ID']
    year = str(START_YEAR - 1 + int(increment)) # Subtract 1 since the task ids start from 1
    
    all_filenames = year_files_dict[year]
    all_gdp_articleids = dict()

    for fname in all_filenames:
        data = utils.load_data(fname)
        all_gdp_articleids[fname] = []
        for art in data['Items']:
            # 1. Filter for english articles only
            if art['data']['language'] == 'en':
                # 2. Filter for articles about the US only
                if US_CODE in art['data']['subjects']:
                    # 3. Filter for articles about inflation only
                    if GDP_CODE in art['data']['subjects']:
                        all_gdp_articleids[fname].append(art['data']['id'])
        print(fname[-15:-8], len(all_gdp_articleids[fname]))
        del data

    with open(TEMP_PATH + '/GDP/GDP_Articles_%s.json' % year, 'w') as f:
        json.dump(all_gdp_articleids, f)



if __name__ == '__main__':
    main() 
