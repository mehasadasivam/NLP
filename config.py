# This file contains all the required paths/ configs for the project

DATA_PATH = '/data/ThomsonReuters_NewsArchive'
TEMP_PATH = '/work/ms5941/NLP/Temp'
OUTPUT_PATH = '/work/ms5941/NLP/Outputs'

# We only want US articles
US_CODE = 'N2:US'

# This are the topic codes that Thomson Reuters already has

INFLATION_CODE = 'N2:INFL'
GDP_CODE = 'N2:GDP'


START_YEAR = 1996
END_YEAR = 2020


TOKENIZED_ARTICLES_PATH = TEMP_PATH + '/%s/%s_Articles_Tokenized_%s.json'

MIN_NUMBER_OF_ARTICLES = 50000
MAX_NUMBER_OF_ARTICLES = 0.7

