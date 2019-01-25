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
    parser.add_argument('--pop_num', type=int, required=True, help='size of population (min 2)')
    parser.add_argument('--prob_cross', type=float, default=0.7, help='crossover probability')
    parser.add_argument('--prob_mutation', type=float, default=0.05, help='mutation probability')
    parser.add_argument('--aggregate', default=False, action='store_true', help='aggregate data groups')
    parser.add_argument('--adapt_function', type=str, default="silhouette",
                        choices=['silh', 'info', 'info_gain', "silhouette"], help='silhouette or information gain')
    parser.add_argument('--dist_measure', type=str, default="euclidean",
                        choices=['eucl', 'manh', 'cos', "euclidean", "manhattan", "cosine"],
                        help='euclidean, manhattan, cosine')
    parser.add_argument('--repeat', type=int, default=1, help='repeat experiment n times and average results')
    parser.add_argument('--logdir', type=str, default="logs", help='aggregate data groups')
    parser.add_argument('--data', type=str, default="train", choices=['train', 'test'], help='aggregate data groups')
    parser.add_argument('--showdata', default=False, action='store_true', help='only show data to be used in experiment')

    args = parser.parse_args(sys_args)
    args.sys_args = sys_args

    if args.pop_num < 2:
        raise TypeError("'pop_num' can't be less than 2")

    if args.dist_measure == 'eucl':
        args.dist_measure = "euclidean"
    elif args.dist_measure == 'manh':
        args.dist_measure = "manhattan"
    elif args.dist_measure == 'cos':
        args.dist_measure = "cosine"
    
    if args.adapt_function == 'silh':
        args.adapt_function = "silhouette"
    elif args.adapt_function == 'info':
        args.adapt_function = "info_gain"

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

    exp_data = hapt_data.get_train_data()
    exp_labs = hapt_data.get_train_labels()
    exp_labels_map = hapt_data.get_labels_map()
    exp_centroids_num = len(hapt_data.get_labels_map())

    if args.data == "test":
        exp_data = hapt_data.get_test_data()
        exp_labs = hapt_data.get_test_labels()
        exp_centroids_num = len(hapt_data.get_labels_map())

    if args.aggregate:
        exp_labs = hapt_data.get_aggregated_train_labels()
        exp_labels_map = hapt_data.get_aggregated_labels_map()
        exp_centroids_num = len(hapt_data.get_aggregated_labels_map())
        if args.data == "test":
            exp_labs = hapt_data.get_aggregated_test_labels()

    # Show experiment data
    # ====================

    if args.showdata:
        utils.plot_clusters(exp_data, exp_labs, exp_labels_map, True)
        return

    # evolution
    # =========
    
    iterations_list, scores_list, populations_list, total_time_list, log_dir_list, best_indiv_idx_list = [],[],[],[],[],[]
    best_overall = (-1, 0, 0, 0) # score, experiment, generation (iteration), individual

    for exp_i in range(args.repeat):
        iterations, scores, populations, total_time, log_dir, best_indiv_idx = evolution.run_SGA(
            args.iter_num, exp_data, exp_labs, args.pop_num, 
            args.prob_cross, args.prob_mutation, exp_centroids_num, 
            args.adapt_function, args.dist_measure, log_dir="logs", loggin_pref="exp {}/{}: ".format(exp_i+1, args.repeat))
        cur_best_score = scores[best_indiv_idx[0], best_indiv_idx[1]]
        if best_overall[0] < cur_best_score:
            best_overall = (cur_best_score, exp_i, best_indiv_idx[0], best_indiv_idx[1])
        
        iterations_list.append(iterations)
        scores_list.append(scores)
        populations_list.append(populations)
        total_time_list.append(total_time)
        log_dir_list.append(log_dir)
        best_indiv_idx_list.append(best_indiv_idx)

        # save plot
        plot_tuple = ("pop:"+str(args.pop_num), "p_c:"+str(args.prob_cross), "p_m:"+str(args.prob_mutation),
                      "data size:"+str(len(exp_labs)), args.adapt_function, args.dist_measure)
        utils.plot_scores(iterations, scores, args.adapt_function, plot_tuple, to_file=True, out_dir=log_dir)

    # visualize
    # =========
    if 1 < args.repeat:
        plot_tuple = ("pop:"+str(args.pop_num), "p_c:"+str(args.prob_cross), "p_m:"+str(args.prob_mutation),
                      "data size:"+str(len(exp_labs)), args.adapt_function, args.dist_measure)
        utils.plot_avg_scores(iterations_list, scores_list, args.adapt_function, best_indiv_idx_list,
            plot_tuple, to_file=True, out_dirs=log_dir_list)

    # new clusteres
    # get best individual

    # correct labels

    # cluster train and test data

    # get confusion matrices, accuracy, measure time

    # plot for train and test data, 
    # utils.plot_clusters(exp_data, exp_labs, exp_labels_map, True)


if __name__ == "__main__":
    main()
