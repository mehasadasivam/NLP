# This contains all the utility methods used to process the 
# Thomson Reuters News Data

from glob import glob

import pandas as pd
import numpy as np
import os

def get_all_filenames():
    folders = glob(os.path.join(DATAPATH, '*'))
    years_files_dict = dict()
    for folder in folders:
        year = folder[:-4]
        years_files_dict[year] = glob(os.path.join(DATAPATH, year, '*'))
    return years_files_dict


