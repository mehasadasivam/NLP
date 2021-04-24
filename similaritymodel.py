#!/apps/anaconda3/bin/python

import os
from config import *
import utilities as utils
import json
import importlib
import nltk
stopwords_list = nltk.corpus.stopwords.words('english')
import re
from gensim.models import Word2Vec

def main():
    year_files_dict = utils.get_all_filenames()
    increment = os.environ['SGE_TASK_ID']
    year = str(START_YEAR - 1 + int(increment)) # Subtract 1 since the task ids start from 1
    
    all_filenames = year_files_dict[year]

    tokenized_sentences = list()

    for fname in all_filenames:

        data = utils.load_data(fname)
        count = 0
        for art in data['Items']:
            text = ''
            if art['data']['language'] == 'en':
                if any(subject in art['data']['subjects'] for subject in [US_CODE]):
                    text = art['data']['body']

            if not text:
                continue
            count += 1
            text = nltk.sent_tokenize(text)
            text = [sentence.replace('\n', ' ') for sentence in text]
            doc_words = []
            for sentence in text:
                tokenized_sent = list()-
                # Only keeping alphabets and replacing everything else with spaces
                sentence = re.sub(r'[^A-Za-z]+', ' ', sentence)
                sentence = sentence.lower()
                sent_words = nltk.word_tokenize(sentence)
                sent_words = [word for word in sent_words if ((len(word) > 1) and (len(word) < 20))]
                sent_words = [word for word in sent_words if word not in stopwords_list]
#                 for word in sent_words:
#                     tokenized_sent.append(word)
                tokenized_sentences.append(sent_words)
        print(fname[-15:-8], 'Done! Article Count:', count, 'Creating word2vec model now.')

    # Build word2vec model and store in temp location
    model = Word2Vec(tokenized_sentences)
    model.save(TEMP_PATH + '/' + 'all_reuters_news_articles_%s.w2v' % (year,))

if __name__ == '__main__':
    main()
