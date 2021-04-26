# This contains all the utility methods used to process the 
# Thomson Reuters News Data

from config import *
from glob import glob

import json
import pandas as pd
import numpy as np
import os

from collections import defaultdict

import gensim
from gensim.test.utils import datapath

import itertools
import re


# Use the natural language toolkit package
import nltk

# Stop Words: Load the nltk default English stopwords list:
stopwords_list = nltk.corpus.stopwords.words('english')



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


# For Topic Model - Pre processing

def get_tokenized_articles(year, theme):
    articles = dict()
    
    with open(TEMP_PATH + '/%s/%s_Article_Texts_v2_%s.json' % (theme, theme, year)) as f:
        yyyymm_all_articles = json.load(f)
    
    for yyyymm in yyyymm_all_articles:
        articles[yyyymm] = []
        for text in yyyymm_all_articles[yyyymm]:
            text.replace('\n', ' ')
            sentences = nltk.sent_tokenize(text)

            text_words = []
            for sentence in sentences:
                sentence = re.sub(r'[^A-Za-z.]+', ' ', sentence)
                sentence = sentence.replace('.', '') # Abbreviations - G.D.P
                sentence = sentence.lower()
                sent_words = nltk.word_tokenize(sentence)
                sent_words = [word for word in sent_words if ((len(word) > 2) and (len(word) < 20))]
                sent_words = [word for word in sent_words if (word not in stopwords_list) and word.isalpha()]
                text_words.extend(sent_words)
            articles[yyyymm].append(text_words)
    
    return articles


def get_effective_vocabulary(articles):
    """
    Articles is a dictionary containing a list of lists for each month.
    
    """
    all_words = itertools.chain.from_iterable(itertools.chain.from_iterable(articles.values()))
    
    # Get frequency counts, sort words by frequency
    frequency_count = nltk.FreqDist(all_words)
    words = np.array([word for word in frequency_count.keys()])
    word_freq = np.array([word for word in frequency_count.values()])
    freq_sort = np.argsort(word_freq)[::-1]
    word_freq_sort = word_freq[freq_sort]
    words_sorted = words[freq_sort]
    
    # Create effective vocabulary: Only keep the words that aren't the 50 most frequent, 
    # and have a frequency of at least 2.
    rank = 1
    effective_vocab = list()
    for object in words_sorted:
        if (rank >= 50):
            fc = frequency_count[object]
            if (fc > 1):
                effective_vocab.append(object)
        rank += 1
    print('Length of effective vocab:', len(effective_vocab))
    return effective_vocab


def get_tokenized_articles_within_effective_vocab(articles):
    effective_vocab = get_effective_vocabulary(articles)
    tok_articles_ev = []
    # Preserve the chronological order in which we are processing articless
    # And lose the dictionary structure
    keys = list(articles.keys())
    keys.sort()
    print(keys)
    for yyyymm in keys:
        for article in articles[yyyymm]:
            article_words_ev = [word for word in article if word in effective_vocab]
            tok_articles_ev.append(article_words_ev)
    return tok_articles_ev


# Filtering dictionaries

# Lists to remove
months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 
          'september', 'october', 'november', 'december', 'jan', 'feb', 'mar', 'apr', 
          'may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'sept', 'daily']
days = ['tuesday', 'day', 'monday', 'thursday', 'wednesday', 'today', 'days', 'friday']
reuters = ['reuterscom', 'reutersnet', 'thomsonreuterscom', 'eikon', 'breakingviews', 'http', 'reuters', 'thomson', 'https', 'www', 'com', 'newsroom']
time = ['gmt', 'years', 'year', 'month', 'months', 'quarter', 'quarters', 'week', 'weekly', 'quarterly', 'monthly', 'yearly']

misc = ['pct', 'would', 'since', 'please', 'per', 'also', 'click', 'first', 'second', 'third', 'fourth', 'inc', 'corp']

def filter_dictionary(theme):
    
    dictionary_all = gensim.corpora.Dictionary.load(TEMP_PATH + '/%s/%s.dict' % (theme, theme))
    print('Length of old dictionary:', len(dictionary_all))
    
    # Filter by thresholds (extreme words)
    MIN_NUMBER_OF_ARTICLES = 50000
    MAX_NUMBER_OF_ARTICLES = 0.7

    dictionary_all.filter_extremes(no_below=MIN_NUMBER_OF_ARTICLES, no_above=MAX_NUMBER_OF_ARTICLES)

    deletable_words = months + days + reuters + time + misc

    # If any of the deletable words remain, store their ids to remove
    del_ids = [k for k,v in dictionary_all.items() if v in deletable_words]

    len(del_ids)

    dictionary_all.filter_tokens(bad_ids=del_ids)

    len(dictionary_all)
    
    dictionary_all.save(TEMP_PATH + '/%s/%s_less_restricted.dict' % (theme, theme))
    print('Clean dictionary saved! New Length: ', len(dictionary_all))
    
