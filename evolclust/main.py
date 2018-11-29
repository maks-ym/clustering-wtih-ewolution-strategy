#!/usr/bin/env python3

import data
import cluster
from matplotlib import pyplot as plt


if __name__ == "__main__":
    # read data
    # convert (FFT)
    # train 
    # predict (test)
    # compare with k-means
    # visualize

    exp_data = data.HAPT()
    

    # print(exp_data.test_data)
    # print(exp_data.test_data.shape)
    # print(exp_data.test_labels)
    # print(len(exp_data.test_data))
    # print(len(exp_data.test_labels))

    data2 = data.HAPT()
    a_ind = 10
    data2.test_data = exp_data.test_data[:a_ind,:5]
    data2.test_labels = exp_data.test_labels[:a_ind]
    print(data2.test_labels)
    print(data2.test_data)


    # train_data = data.HAPT("data/hapt/")
    # train_data.load_train("data/hapt/train/")

    # print(train_data.train_data)
    # print(train_data.train_labels)
    # print(len(train_data.train_data))
    # print(len(train_data.train_labels))