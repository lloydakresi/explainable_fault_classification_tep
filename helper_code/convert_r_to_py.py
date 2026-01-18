import pyreadr
import pandas as pd
import numpy as np

def r_2_py(file_names):
    '''
    Converts a list of rdata files into pandas dataframes

    :param file_names: list of file names

    '''
    for name in file_names:
        ordered_dict = pyreadr.read_r(f'{name}')
        key = list(ordered_dict.keys())[0]
        df = ordered_dict[key]
        df.to_pickle(f"{key}.pkl")


files = ['dataset_r/TEP_FaultFree_Testing.RData',
         'dataset_r/TEP_FaultFree_Training.RData',
         'dataset_r/TEP_Faulty_Testing.RData',
         'dataset_r/TEP_Faulty_Training.RData']

r_2_py(files)
