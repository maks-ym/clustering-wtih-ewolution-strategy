#!/usr/bin/env python3

# classes to work with different datasets

import numpy as np
import os
import utils


class HAPT:
    def __init__(self, dir_path="data/hapt/"):
        self.data_dir_path = dir_path
        self._train_attrs = None
        self._test_attrs = None
        self._train_labels = None
        self._test_labels = None
        self._labels = {}
        self._aggregated_labels = {}
        self._aggregated2initial_labels = {}
        self._aggregated_test_labels = None
        self._aggregated_train_labels = None
        self._aggregated_labels = {0: "WALKING", 1: "STATIC", 2: "TRANSITION"}
        self._aggregation_map = {
            1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1,
            7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2
        }

    def get_labels_map(self):
        '''
        get names for activity labels (int to string)
        '''
        if not self._labels:
            file_path = os.path.join(self.data_dir_path, "activity_labels.txt")
            with open(file_path) as l_file:
                for line in l_file:
                    line_arr = line.strip("\n ").split(" ")
                    self._labels[int(line_arr[0])] = line_arr[1]
        return self._labels

    def get_train_data(self):
        '''
        return numpy array of attributes, shape: (samples_num, feats_num)
        '''
        if self._train_attrs is None:
            file_path = os.path.join(self.data_dir_path, "train/x_train.txt")
            self._train_attrs = np.loadtxt(file_path)
        return self._train_attrs

    def get_test_data(self):
        '''
        return numpy array of attributes, shape: (samples_num, feats_num)
        '''
        if self._test_attrs is None:
            file_path = os.path.join(self.data_dir_path, "test/x_test.txt")
            self._test_attrs = np.loadtxt(file_path)
        return self._test_attrs

    def get_train_labels(self):
        if self._train_labels is None:
            file_path = os.path.join(self.data_dir_path, "train/y_train.txt")
            self._train_labels = np.loadtxt(file_path, dtype=int)
        return self._train_labels

    def get_test_labels(self):
        if self._test_labels is None:
            file_path = os.path.join(self.data_dir_path, "test/y_test.txt")
            self._test_labels = np.loadtxt(file_path, dtype=int)
        return self._test_labels

    def load_train_data(self):
        self.get_train_data()
        self.get_train_labels()
        if self._train_attrs is None:
            raise ValueError("_train_attrs failed to load")
        if self._train_labels is None:
            raise ValueError("_train_labels failed to load")
        if len(self._train_attrs) != len(self._train_labels):
            raise ValueError("train samples num != train labels num")

    def load_test_data(self):
        self.get_test_data()
        self.get_test_labels()
        if self._test_attrs is None:
            raise ValueError("_test_attrs failed to load")
        if self._test_labels is None:
            raise ValueError("_test_labels failed to load")
        if len(self._test_attrs) != len(self._test_labels):
            raise ValueError("test samples num != test labels num")

    def load_all_data(self):
        self.load_train_data()
        self.load_test_data()
        self.get_labels_map()

    def aggregate_groups(self):
        self.get_labels_map()
        if self._test_labels is None:
            print("WARNING: No test labels. No aggregation.")
            return
        if self._train_labels is None:
            print("WARNING: No train labels. No aggregation.")
            return
        if self._labels == {}:
            print("WARNING: No labels map. No aggregation.")
            return

        for key in self._aggregation_map:
            if self._aggregation_map[key] not in self._aggregated2initial_labels:
                self._aggregated2initial_labels[self._aggregation_map[key]] = []
            self._aggregated2initial_labels[self._aggregation_map[key]].append(key)

        self._aggregated_test_labels = [self._aggregation_map[l] for l in self._test_labels]
        self._aggregated_train_labels = [self._aggregation_map[l] for l in self._train_labels]
        self._aggregated_test_labels = np.array(self._aggregated_test_labels)
        self._aggregated_train_labels = np.array(self._aggregated_train_labels)

    def get_aggr2initial_labs_map(self):
        return {self._aggregated_labels[k]: [self._labels[l] for l in v_list]
                for k, v_list in self._aggregated2initial_labels.items()}

    def get_aggregated_test_labels(self):
        if self._aggregated_test_labels is None:
            print("WARNING: Data are not aggregated yet! Initial labels returned")
            return self._test_labels
        return self._aggregated_test_labels

    def get_aggregated_train_labels(self):
        if self._aggregated_train_labels is None:
            print("WARNING: Data are not aggregated yet! Initial labels returned")
            return self._train_labels
        return self._aggregated_train_labels

    def get_aggregated_labels_map(self):
        return self._aggregated_labels
