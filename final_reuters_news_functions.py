from collections import Counter
from datetime import datetime
from glob import glob
from nltk.stem.snowball import SnowballStemmer

from config import *

import dateutil
import json
import matplotlib.pyplot as plt
import pandas as pd
import re


def get_reuters_news_files():
    return glob(os.path.join(REUTERS_PATH, '*', '*'))


def load_data(filename):
    fp = open(filename)
    data = json.load(fp)
    fp.close()
    return data


def get_gd_articles(news_files):
    gd_articles = []
    for file in news_files:
        data = load_data(file)
        for art in data['Items']:
            # Conditions
            # First: Language must be English
            english = art['data']['language'] == 'en'
            
            # Second: Should not be an alert (Urgency != 1)
            not_alert = art['data']['urgency'] != 1
            
            # Third: Contains the term 'Great Depression'
            # Allow any lower case occurences of the exact term,
            # and check for cases where an endline character comes in between
            contains_great_depression = 'great depression' in art['data']['body'].lower().replace('\n', ' ')
            
            if english and not_alert and contains_great_depression:
                gd_articles.append(art)
    return gd_articles


def article_hours(articles):
    plt.figure(figsize=FIG_SIZE)
    hrs = [art['timestamps'][0]['timestamp'].hour for art in articles]
    plt.hist(hrs, rwidth=0.9, bins=24)
    plt.suptitle('Article hour distribution for ' + '{:,}'.format(len(hrs)) + ' articles (NY Time)')
    plt.xlabel("Day")
    plt.ylabel("Number of Articles")
    plt.show()


def convert_to_ny_timestamps(gd_articles):
    def ny_time(time_str):
        utc_t = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S.%f%z')
        ny_t = utc_t.astimezone(dateutil.tz.gettz('America/New_York'))
        return ny_t

    for art in gd_articles:
        art['timestamps'][0]['timestamp'] = ny_time(art['timestamps'][0]['timestamp'])

    return gd_articles


def get_vix_data():
    vix = pd.read_csv(VIX_DATA_FILE)
    vix.Date = pd.to_datetime(vix.Date)
    vix = vix[vix.Date >= pd.to_datetime('2008-09-01')]
    vix = vix.set_index('Date')
    return vix


def _article_days(articles):
    days = [art['timestamps'][0]['timestamp'].date() for art in articles]
    return days


def plot_article_days_hist(gd_articles):
    plt.figure(figsize=FIG_SIZE)
    days = _article_days(gd_articles)
    plt.hist(days, rwidth=0.7, bins=len(set(days)))
    plt.suptitle('Article day distribution for ' + '{:,}'.format(len(days)) + ' articles')
    plt.xlabel("Day")
    plt.ylabel("Number of Articles")
    plt.show()


def vix_article_days_combined_plot(gd_articles):
    """
    Plots a single graph with both the article count and the VIX during 
    the same time period (Sept 2008 - March 2009) to compare trends
    
    """
    vix = get_vix_data()

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    # Plot the day article count
    days = _article_days(gd_articles)
    ax.hist(days, rwidth=0.7, color='orange', bins=len(set(days)))
    ax.set_xlabel("Day")
    ax.set_ylabel("Number of Articles")

    # Twin object for two different y-axis on the sample plot
    ax2 = ax.twinx()
    # Make a plot for VIX with different y-axis using twin axis object
    ax2.plot(vix.index, vix['Close'])
    ax2.set_ylabel("VIX Close")
    plt.grid()
    plt.show()


def get_terms(text):
    """
    Return a list of lower cases, stemmed terms from the input text (without stopwords)

    """
    stemmer = SnowballStemmer('english')

    # Remove all non-alphanumeric characters, replace them with ' '
    text = re.sub(r'\W+', ' ', text)
    # Convert to lower case, split the string by ' ' into a list of terms
    text = text.lower().split()
    # Remove stopwords from the nltk list,
    # and also remove the single character words
    text = [word for word in text if word not in stopwords_list]
    text = [word for word in text if len(word) > 1]
    # Stem the remaining words
    text = [stemmer.stem(wrd) for wrd in text]
    return text


def construct_dtm(articles):
    dtm_rows = []
    for art in articles:
        terms = get_terms(art['data']['body'])
        dtm_row = Counter(terms)
        data_as_list = [(art['guid'], wrd, cnt) for wrd, cnt in dtm_row.items()]
        dtm_rows.extend(data_as_list)

    df = pd.DataFrame(dtm_rows, columns=['Id', 'Word', 'Count'])
    return df


def get_word_frequencies(dtm):
    # Count the number of rows (grouped by words) to create a frequency df
    freq_df = dtm.groupby('Word').Count.sum().reset_index()
    # Sort the rows based on Count, in descending order
    freq_df = freq_df.sort_values('Count', ascending=False).reset_index(drop=True)

    return freq_df
