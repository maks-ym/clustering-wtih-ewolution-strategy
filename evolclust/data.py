#!/usr/bin/env python3

# classes to work with different datasets

import numpy as np
import os
import time
import utils

class HAPT:
    def __init__(self, dir_path="data/hapt/"):
        self.data_dir_path = dir_path
        self._train_attrs = []
        self._test_attrs = []
        self._train_labels = []
        self._test_labels = []
        self._labels = {}# map of labels
        self._statistics = {}

    @utils.timeit
    def get_train_feats(self):
        '''
        return numpy array of attributes, shape: (samples_num, feats_num)
        '''
        if not self._train_attrs:
            file_path = os.path.join(self.data_dir_path, "train/x_train.txt")
            self._train_attrs = np.loadtxt(file_path)
        return self._train_attrs


    # TODO clean function
    @utils.timeit
    def get_test_feats(self):
        '''
        return numpy array of attributes, shape: (samples_num, feats_num)
        '''
        if not self._train_attrs:
            file_path = os.path.join(self.data_dir_path, "test/x_test.txt")
            self._test_attrs = np.loadtxt(file_path)
        return self._test_attrs


    def get_train_labels(self):
        if not self._train_labels:
            file_path = os.path.join(self.data_dir_path, "train/y_train.txt")
            self._train_labels = np.loadtxt(file_path, dtype=int)
        return self._train_labels

    
    def get_test_labels(self):
        if not self._test_labels:
            file_path = os.path.join(self.data_dir_path, "test/y_test.txt")
            self._test_labels = np.loadtxt(file_path, dtype=int)
        return self._test_labels


    def load_train_data(self):
        self.get_train_feats
        self.get_train_labels
        if len(self._train_attrs) != len(self._train_labels):
            raise ValueError("train samples num != train labels num")


    def load_test_data(self):
        self.get_test_feats
        self.get_test_labels
        if len(self._test_attrs) != len(self._test_labels):
            raise ValueError("test samples num != test labels num")


    def load_all_data(self):
        self.load_test_data
        self.load_train_data
    