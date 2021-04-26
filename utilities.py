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


def kern(x):
    """kernel function"""
    return np.power((1-np.power(x,3)),3)


def standarized(variable):
    mean = np.cumsum(variable)/np.arange(1, 26)
    std_dev = np.sqrt((np.cumsum(np.power(variable,2)) - np.power(mean,2)*np.arange(1, 26))/np.arange(0, 25))
    
    std_variable = (variable - mean)/std_dev
    
    return std_variable


def distance(distance_inputs, ref_period):
    distance = np.zeros(len(distance_inputs[0]))
    dist_int = list()
    
    for j in range(len(distance_inputs)):
        distance_process = distance_inputs[j] - distance_inputs[j][ref_period]
        distance_process = np.power(distance_process,2)
        dist_int.append(distance_process)
    
    for i in range(len(dist_int)):
        distance = distance + dist_int[i]
    distance = np.power(distance,0.5)
    
    return distance


# Filtering dictionaries and creating new corpuses

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
    dictionary_all.filter_extremes(no_below=MIN_NUMBER_OF_ARTICLES, no_above=MAX_NUMBER_OF_ARTICLES)

    # Updated stop words
    deletable_words = months + days + reuters + time + misc

    # If any of the deletable words remain, store their ids to remove
    del_ids = [k for k,v in dictionary_all.items() if v in deletable_words]

    len(del_ids)

    dictionary_all.filter_tokens(bad_ids=del_ids)

    len(dictionary_all)
    
    dictionary_all.save(TEMP_PATH + '/%s/%s_less_restricted.dict' % (theme, theme))
    print('Clean dictionary saved! New Length: ', len(dictionary_all))
    

def create_new_corpuses(theme, end_year=None):
    if not end_year:
        years = [str(year) for year in range(START_YEAR, END_YEAR + 1)]
    else:
        years = [str(year) for year in range(START_YEAR, end_year + 1)]
    print(years)

    all_tok_articles = []
    for year in years:
        with open(TOKENIZED_ARTICLES_PATH % (theme, theme, year)) as f:
            all_tok_articles.extend(json.load(f))
        print(TOKENIZED_ARTICLES_PATH % (theme, theme, year), 'done!')
    
    dictionary_all = gensim.corpora.Dictionary.load(TEMP_PATH + '/%s/%s_less_restricted.dict' % (theme, theme))

    class MyCorpus:
        def __iter__(self):
            for doc in all_tok_articles:
                # assume there's one document per line, tokens separated by whitespace
                yield dictionary_all.doc2bow(doc)

    corpus_memory_friendly = MyCorpus()  # doesn't load the corpus into memory!
    print(corpus_memory_friendly)

    gensim.corpora.MmCorpus.serialize(TEMP_PATH + '/%s/%s_less_restricted.mm' % (theme, theme), corpus_memory_friendly)
    
    
# Regular LDA Utilities

def generate_lda_model(theme, corpus, dictionary, num_topics=15, passes=25, 
                       iterations=400, eval_every=None, update_every=0, 
                       alpha='auto', eta='auto'):

    lda = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, alpha='auto', eta='auto',
                                 iterations=iterations, num_topics=num_topics, passes=passes, 
                                 eval_every=eval_every, update_every = update_every)
    
    # Save lda model
    tempfile = TEMP_PATH + '/%s/%s_LDA_model_' % (theme, theme) + '_'.join([str(num_topics), str(passes), str(iterations), str(alpha), str(eta)]) 
    lda.save(tempfile)
    
    return lda


def get_model(theme, corpus_all, dictionary_all, num_topics=15, passes=25, iterations=400, 
              eval_every=None, update_every=0, alpha='auto', eta='auto'):
    """
    Get the LDA model 
    
    """
    # Check if a model with the same config already exists. 
    # If it does, load the model instead of generating a new one
    tempfile = TEMP_PATH + '/%s/%s_LDA_model_' % (theme, theme) + '_'.join([str(num_topics), str(passes), str(iterations), str(alpha), str(eta)])

    if os.path.exists(tempfile):
        lda = gensim.models.LdaModel.load(tempfile)
    else:
        lda = generate_lda_model(theme, corpus_all, dictionary_all, num_topics, passes, 
                                 iterations, eval_every, update_every, alpha, eta)
    return lda



def get_avg_topic_probabilities(lda, corp, num_topics):
    """
    For the given LDA model and corpus, get the aggregate probability of each topic 
    (by iterating over each article in the year's corpus, adding up individual probabilities)
    Then, divide by the total number of articles for the year to get the average 
    topic probabilities for the corpus.
    
    """
    all_topics_probabilities = np.zeros(num_topics)
    for article in corp:
        article_topics = lda.get_document_topics(article)
        topic_vec = np.zeros(num_topics)
        for k, prob in article_topics:
            topic_vec[k] = prob
        all_topics_probabilities += topic_vec
    
    # Avg topic probabilities
    avg_topic_probabilities = all_topics_probabilities/float(len(corp))
    
    return avg_topic_probabilities


def get_top_ten_topics_for_year(year, lda, avg_topic_probabilities):
    """
    Using the average topic probabilites, rank the topics and 
    return the top ten topics for a year.
    
    """
    # Get top 10 topics for each year
    indices = (-avg_topic_probabilities).argsort()[:10]

    top_topics_words = dict()
    top_topics_words[year] = dict()
    rank = 1
    for ind in indices:
        top_words = lda.show_topic(ind, topn=10)
        words, probs = zip(*top_words)
        top_topics_words[year][rank] = top_words
        rank += 1

    df = pd.DataFrame.from_dict({(i,j): [x[0] for x in top_topics_words[i][j]] for i in top_topics_words.keys() 
                            for j in top_topics_words[i].keys()}).T
    return df, top_topics_words


def get_topics(theme, corpus_all, dictionary_all, corpus_by_year, num_topics=15, passes=25, iterations=400, 
               eval_every=None, update_every=0, alpha='auto', eta='auto'):
    """
    Get the top topics for each year, based on an LDA model created using articles across all years
    
    """
    lda = get_model(theme, corpus_all, dictionary_all, num_topics, passes, 
                    iterations, eval_every, update_every, alpha, eta)

    avg_topics_all = []
    for year in range(START_YEAR, END_YEAR + 1):
        avg_topic_probabilities = get_avg_topic_probabilities(lda, corpus_by_year[year], num_topics)
        df, top_topic_words = get_top_ten_topics_for_year(year, lda, avg_topic_probabilities)
        display(df)
        avg_topics_all.append(avg_topic_probabilities)

    return avg_topics_all


