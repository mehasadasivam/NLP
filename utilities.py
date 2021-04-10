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


def load_data(filename):
    fp = open(filename)
    try:
        data = json.load(fp)
    except:
        print(filename, 'too big. Handle differently and not through conventional json loading.')
        data = None
    fp.close()
    return data


def filtered_article_count(data, language='en', subjects=None, subject_filter_type='any'):
    if subjects is None:
        return language_count(data, language)
    count = 0
    for art in data['Items']:
        if art['data']['language'] == language:
            if eval(subject_filter_type)(subject in art['data']['subjects'] for subject in subjects):
                count +=1
    return count
