import cluster
import data
import evolution
import utils

from ast import literal_eval as make_tuple
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

import argparse
import os
import sys


def parse_arguments(sys_args):
    """
    Parse, check and prepare arguments for main script.

    :param sys_args: arguments passed to script without script name
    :return:         processed args object if succeed
    """
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--path', type=str, required=True, help='path to logs after experiment')
    parser.add_argument('--outdir', type=str, default=None, help='directory for inference results output')
    parser.add_argument('--nooutput', default=False, action='store_true', help='only show results and do not output to files')

    args = parser.parse_args(sys_args)
    args.sys_args = sys_args

    if args.outdir and args.nooutput:
        raise TypeError("'outdir' and 'nooutput' cannot be used at the same time")

    if args.outdir is None:
        args.outdir = os.path.join(args.path, "inference_output")
    else:
        args.outdir = os.path.join(args.outdir, "inference_output")

    return args



def main():
    args = parse_arguments(sys.argv[1:])

    # read params
    # ===========
    # possible params:
    # iter_num, pop_num, centers_num, prob_cross, prob_mutation, data shape, labs shape, 
    # adapt_function, dist_measure, log_dir, best score, best score (index), total_time

    exp_params = {}
    text_file = [f for f in os.listdir(args.path) if f.endswith(".txt")][0]
    with open(os.path.join(args.path, text_file), "r") as text_f:
        for line in text_f:
            line = line.replace("\t", "").strip().split(":")
            if len(line) == 2 and line[0] != "" and line[1] != "":
                if line[0] == "iter_num" or line[0] == "pop_num" or line[0] == "centers_num":
                    exp_params[line[0].replace(" ", "_")] = int(line[1])
                elif line[0] == "prob_cross" or line[0] == "prob_mutation" or line[0] == "best score":
                    exp_params[line[0].replace(" ", "_")] = float(line[1])
                elif line[0] == "data shape" or line[0] == "labs shape":
                    exp_params[line[0].replace(" ", "_")] = make_tuple(line[1])
                elif line[0] == "best score (index)":
                    #best score (index):	generation 95, individual 99
                    line[1] = line[1].strip().split(",")
                    exp_params["best_index"] = (int(line[1][0].strip().split(" ")[1]), int(line[1][1].strip().split(" ")[1]))
                else:
                    exp_params[line[0].replace(" ", "_")] = line[1]

    print("\nexperiment parameters were:")
    for k, v in exp_params.items():
        print("{:20}: {}".format(k, v))


    # read results
    # ============

    generations = np.load(os.path.join(args.path, "generations.npy"))
    iterations = np.load(os.path.join(args.path, "iterations.npy"))
    scores = np.load(os.path.join(args.path, "scores.npy"))

    best_centers = generations[exp_params["best_index"][0], exp_params["best_index"][1]]

    print("\nobtained results are:")
    print("generations (total num, pop size, centrs num, feats num): {}".format(generations.shape))
    print("iterations (iterations num, ):                            {}".format(iterations.shape))
    print("scores (total num, pop size):                             {}".format(scores.shape))
    print("generations total num, iterations num and scores total num must be equal!")
    print("generations pop size and scores pop size must be equal too!")

    plot_tuple = ("pop:"+str(exp_params["pop_num"]), "p_c:"+str(exp_params["prob_cross"]), "p_m:"+str(exp_params["prob_mutation"]),
                      "data size:"+str(len(exp_params["data_shape"])), exp_params["adapt_function"], exp_params["dist_measure"], "best score:"+str(exp_params["best_score"])[:9]+" at "+str(exp_params["best_index"]))
    utils.plot_scores(iterations, scores, exp_params["adapt_function"], plot_tuple, not args.nooutput, out_dir=args.outdir)

    # read data
    # =========
    print("reading data...")
    hapt_data = data.HAPT()
    hapt_data.load_all_data()
    hapt_data.aggregate_groups()

    test_data = hapt_data.get_test_data()
    test_labs = hapt_data.get_test_labels()
    train_data = hapt_data.get_train_data()
    train_labs = hapt_data.get_train_labels()
    labs_map = hapt_data.get_labels_map()
    if exp_params["centers_num"] == 3:
        test_labs = hapt_data.get_aggregated_test_labels()
        train_labs = hapt_data.get_aggregated_train_labels()
        labs_map = hapt_data.get_aggregated_labels_map()
    centroids_num = len(labs_map)


    assert exp_params["centers_num"] == centroids_num

    # do clusterizations
    # ==================
    print("clustering...")
    labels_names = list(labs_map.values())
    # train data
    train_clust_labs = cluster.Centroids.cluster(train_data, best_centers, dist_func=exp_params["dist_measure"])
    train_clust_labs = cluster.Utils.adjust_labels(train_clust_labs, train_labs)
    train_silh = cluster.Evaluate.silhouette(train_data, train_clust_labs, exp_params["dist_measure"])
    train_silh_normalized = (train_silh + 1) / 2
    train_info_gain = cluster.Evaluate.information_gain(train_labs, train_clust_labs)
    mapped_train_clust_labs = [labs_map[l] for l in train_clust_labs]
    mapped_train_labs = [labs_map[l] for l in train_labs]
    train_conf_mtx = confusion_matrix(mapped_train_labs, mapped_train_clust_labs, labels=labels_names)
    print("train set\tsilh: {:.6}, silh normalized: {:.6}, info gain: {:.6}".format(train_silh, train_silh_normalized, train_info_gain))
    # test data
    test_clust_labs = cluster.Centroids.cluster(test_data, best_centers, dist_func=exp_params["dist_measure"])
    test_clust_labs = cluster.Utils.adjust_labels(test_clust_labs, test_labs)
    test_silh = cluster.Evaluate.silhouette(test_data, test_clust_labs, exp_params["dist_measure"])
    test_silh_normalized = (test_silh + 1) / 2
    test_info_gain = cluster.Evaluate.information_gain(test_labs, test_clust_labs)
    mapped_test_clust_labs = [labs_map[l] for l in test_clust_labs]
    mapped_test_labs = [labs_map[l] for l in test_labs]
    test_conf_mtx = confusion_matrix(mapped_test_labs, mapped_test_clust_labs, labels=labels_names)
    print("test set\tsilh: {:.6}, silh normalized: {:.6}, info gain: {:.6}".format(test_silh, test_silh_normalized, test_info_gain))


    # Show data
    # =========
    print("creating plots...")
    # clusters
    utils.plot_clusters(train_data, train_labs,       labs_map, True, out_dir=args.outdir, filename="train_orig_clusters")
    utils.plot_clusters(train_data, train_clust_labs, labs_map, True, out_dir=args.outdir, filename="train_obtained_clusters")
    utils.plot_clusters(test_data,  test_labs,        labs_map, True, out_dir=args.outdir, filename="test_orig_clusters")
    utils.plot_clusters(test_data,  test_clust_labs,  labs_map, True, out_dir=args.outdir, filename="test_obtained_clusters")

    # confusion matrices
    utils.plot_confusion_matrix(train_conf_mtx, labels_names, normalize=False, 
            title='Confusion matrix\ntrain set\n(silh: {:.6}, silh normalized: {:.6}, info gain: {:.6})'.format(train_silh, train_silh_normalized, train_info_gain),
            cmap=plt.cm.Blues, out_dir=args.outdir, filename="train_conf_matr_silh_info_gain")
    utils.plot_confusion_matrix(test_conf_mtx, labels_names, normalize=False, 
            title='Confusion matrix\ntest set\n(silh: {:.6}, silh normalized: {:.6}, info gain: {:.6})'.format(test_silh, test_silh_normalized, test_info_gain),
            cmap=plt.cm.Blues, out_dir=args.outdir, filename="train_conf_matr_silh_info_gain")
    print("inference ended")

if __name__ == "__main__":
    main()
