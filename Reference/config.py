import os
import nltk

# Paths to files
PATH          = '/Users/mehasadasivam/Courses/Final Exam/'
RE_DATA_PATH  = os.path.join(PATH, 'real_estate_data_sample.csv')
EDGAR_PATH    = os.path.join(PATH, 'edgar100')
MODEL_NAME    = 'CBOW_window_%d_vecsize_%d.w2v'
MODEL_PATH    = os.path.join(PATH, MODEL_NAME)
REUTERS_PATH  = os.path.join(PATH, 'reuters_news')
VIX_DATA_FILE = os.path.join(PATH, 'VIX_YahooFinance.csv')

# Load the stopwords from the nltk corpus
stopwords_list = nltk.corpus.stopwords.words('english')

# Adding some more stop words to the list
stopwords_list.append('would')
stopwords_list.append('could')
stopwords_list.append('said')
stopwords_list.append('id')
stopwords_list.append('com')
stopwords_list.append('quot')

# Other parameters for consistency:
FIG_SIZE = (15, 5)
