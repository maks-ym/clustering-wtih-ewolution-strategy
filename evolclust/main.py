#!/usr/bin/env python3

import cluster
import data
import evolution
import utils

from matplotlib import pyplot as plt
import argparse
import sys


def parse_arguments(sys_args):
    """
    Parse, check and prepare arguments for main script.

    :param sys_args: arguments passed to script without script name
    :return:         processed args object if succeed
    """
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--iter_num', type=int, required=True, help='number of iterations')
    parser.add_argument('--pop_num', type=int, required=True, help='size of population')
    parser.add_argument('--prob_cross', type=float, default=0.7, help='crossover probability')
    parser.add_argument('--prob_mutation', type=float, default=0.05, help='mutation probability')
    parser.add_argument('--aggregate', default=False, action='store_true', help='aggregate data groups')
    parser.add_argument('--adapt_function', type=str, default="silhouette",
                        choices=['silh', 'info', 'info_gain', "silhouette"], help='silhouette or information gain')
    parser.add_argument('--dist_measure', type=str, default="euclidean",
                        choices=['eucl', 'manh', 'cos', "euclidean", "manhattan", "cosine"],
                        help='euclidean, manhattan, cosine')
    parser.add_argument('--logdir', type=str, default="logs", help='aggregate data groups')
    parser.add_argument('--data', type=str, default="train", choices=['train', 'test'], help='aggregate data groups')

    args = parser.parse_args(sys_args)
    args.sys_args = sys_args

    if args.dist_measure == 'eucl':
        args.dist_measure = "euclidean"
    elif args.dist_measure == 'manh':
        args.dist_measure = "manhattan"
    elif args.dist_measure == 'cos':
        args.dist_measure = "cosine"
    
    if args.adapt_function == 'silh':
        args.adapt_function = "silhouette"

    # check, fix, add arguments
    return args


def main():
    args = parse_arguments(sys.argv[1:])

    print("Parameters:")
    for arg_ in args.sys_args:
        print(arg_)
    print()

    # read data
    # =========
   
    hapt_data = data.HAPT()
    hapt_data.load_all_data()
    hapt_data.aggregate_groups()

    # Plot raw test set
    # if args.showdata:
        # utils.plot_clusters(data2.test_data, data2.test_labels, data2.get_labels_map())
        # utils.plot_clusters_3d(data2.test_data, data2.test_labels, data2.get_labels_map())

    # plot aggregated test set
    # if args.showdata and args.aggregate:
        # utils.plot_clusters(data2.get_train_data(), data2.get_aggregated_train_labels(), data2.get_aggregated_labels_map())
        # utils.plot_clusters(data2.get_test_data(), data2.get_aggregated_test_labels(), data2.get_aggregated_labels_map())
        # utils.plot_clusters_3d(data2.get_test_data(), data2.get_aggregated_test_labels(), data2.get_aggregated_labels_map())

    exp_data = hapt_data.get_train_data()
    exp_labs = hapt_data.get_train_labels()
    exp_centroids_num = len(hapt_data.get_labels_map())

    if args.data == "test":
        exp_data = hapt_data.get_test_data()
        exp_labs = hapt_data.get_test_labels()
        exp_centroids_num = len(hapt_data.get_labels_map())

    if args.aggregate:
        exp_labs = hapt_data.get_aggregated_train_labels()
        exp_centroids_num = len(hapt_data.get_aggregated_labels_map())
        if args.data == "test":
            exp_labs = hapt_data.get_aggregated_test_labels()

    # evolution
    # =========
    iterations, scores, generations, total_time, log_dir = evolution.run_SGA(
            args.iter_num, exp_data, exp_labs, args.pop_num, 
            args.prob_cross, args.prob_mutation, exp_centroids_num, 
            args.adapt_function, args.dist_measure, log_dir="logs")

    # predict (test)
    # compare with ground truth

    # visualize
    # =========
    utils.plot_scores(iterations, scores, args.adapt_function, 
            (args.pop_num, args.prob_cross, args.prob_mutation, len(exp_labs), 
            args.adapt_function, args.dist_measure), to_file=True, out_dir=log_dir)

if __name__ == "__main__":
    main()
