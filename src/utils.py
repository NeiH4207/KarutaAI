'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
from threading import Thread
import time
import numpy as np
import torch
from random import seed

seed(2)

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

def get_filename(filepath):
    return os.path.basename(filepath).split(".")[0]

def gather_files_from_folder(_dir, _extension):
    """
    :param _dir: Directory where file of interest are stored.
    :param _extension: Extension of file of interest.
    :return: A list of files.
    """
    list_files = list()
    for root, dirs, files in os.walk(_dir):
        for file in files:
            if file.endswith(_extension):
                list_files.append(os.path.join(root, file))
    return list_files
    
def gather_files(_dir, _extension, _list, isdir = False):
    """

    :param _dir: Directory where file of interest are stored.
    :param _extension: Extension of file of interest.
    :param _list: List of files of interest. Look for those files in _dir.
    :param pattern: In some case we need to look for a pattern and not for an extension.
    :return: A list of files.

    Regroup files of interest from a directory and all the subdirectories.
    """
    check_list = list()
    if isdir is False:

        for prefix in _list:
            if os.path.isfile(os.path.join(_dir, prefix) + _extension):
                check_list.append(os.path.join(_dir,prefix) + _extension)
    else:
        for prefix in _list:
            if os.path.isdir(os.path.join(_dir, prefix) + _extension):
                check_list.append(os.path.join(_dir, prefix) + _extension)

    # return list_files
    return check_list
