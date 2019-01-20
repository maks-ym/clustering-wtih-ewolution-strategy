#!/usr/bin/env python3

import data
import cluster
from matplotlib import pyplot as plt
import utils


if __name__ == "__main__":
    # read data
    # train 
    # predict (test)
    # compare with ground truth
    # visualize

    data2 = data.HAPT()
    data2.load_all_data()
    a_ind = 10

    # Plot not raw test set
    # utils.plot_clusters(data2.test_data, data2.test_labels, data2.get_labels_map())
    # utils.plot_clusters_3d(data2.test_data, data2.test_labels, data2.get_labels_map())

    data2.aggregate_groups()
    # plot aggregated test set
    utils.plot_clusters(data2.get_train_data(), data2.get_aggregated_train_labels(), data2.get_aggregated_labels_map())
    utils.plot_clusters(data2.get_test_data(), data2.get_aggregated_test_labels(), data2.get_aggregated_labels_map())
    # utils.plot_clusters_3d(data2.get_test_data(), data2.get_aggregated_test_labels(), data2.get_aggregated_labels_map())
