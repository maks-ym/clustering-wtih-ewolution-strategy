"""
Simple Genetic Algorithm
"""
import cluster
import numpy as np
import time
from datetime import datetime

import os
import logging

logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)

def init_population(data, centers_num=12, n=20):
    """
        Choose randomly genes for chromosomes between minima and maxima of 
        corresponding genes in data

        data        : data to get minima and maxima for generation
        centers_num : number of centroids per individual to generate
        n           : number of indiwiduals

        return : initial population pop_0, maxima and minima for mutation
    """
    logging.info("init_population - start")
    data = np.array(data)
    maxima = data.max(axis=0)
    minima = data.min(axis=0)
    logging.debug("data maxima:\t{}".format(maxima))
    logging.debug("data minima:\t{}".format(minima))
    logging.debug("data maxima len:\t{}".format(len(maxima)))
    pop = (maxima - minima) * np.random.random_sample((n, centers_num, len(data[0]))) + minima
    logging.debug("init_population - end")
    return pop, maxima, minima


def get_adapt_scores(pop_t, data, true_labs, adapt_function="silhouette", dist_measure="euclidean"):
    """
        Cluster data for eacn individual (centroids set) in population and get scores.

        pop_t           : population to score
        true_labs       : for data provided
        adapt_function  : {"silhouette"|"info_gain"}; these values to be returned
        dist_measure    : {"euclidean"|"manhattan"|"cosine"} used for clustering and silhouette score

        return : scores for each individual; 1-D array of length n (pop_t size)
    """
    # logging.info("getting adapt scores...")
    logging.debug("pop_t len:\t{}".format(len(pop_t)))
    logging.debug("data len:\t{}".format(len(data)))
    logging.debug("adapt_function:\t{}".format(adapt_function))
    logging.debug("dist_measure:\t{}".format(dist_measure))

    uniq_true = np.unique(true_labs)
    # TODO: vectorize
    scores = []
    for i, individual in enumerate(pop_t):
        logging.debug("{}: Clustering...".format(i))

        labs = cluster.Centroids.cluster(data, individual, dist_func=dist_measure)
        unique_labs = np.unique(labs)
        logging.debug("{}: labs : {}".format(i, labs))

        logging.debug("{}: labs unique : {}".format(i, unique_labs))
        if adapt_function == "silhouette":
            if len(unique_labs) < len(uniq_true):
                cur_score = -1
            else:
                logging.debug("{}: computing silhouette...".format(i))
                cur_score = cluster.Evaluate.silhouette(data, labs, dist_func=dist_measure)
            logging.debug("{}: cur_score: {}".format(i, cur_score))
            scores.append(cur_score)
        elif adapt_function == "info_gain":
            logging.debug("{}: computing info_gain...".format(i))
            labs = cluster.Utils.adjust_labels(labs, true_labs)
            logging.debug("{}: adjust_labels: {}".format(i, labs))
            logging.debug("{}: adjust_labels length: {}".format(i, len(labs)))
            cur_score = cluster.Evaluate.information_gain(true_labs, labs)
            logging.debug("{}: cur_score: {}".format(i, cur_score))
            scores.append(cur_score)
    
    logging.debug("scores: {}".format(scores))
    logging.debug("DONE - getting adapt scores")
    return scores


def reproduction(pop_t, adapt_scores):
    """
        Randomly copy individuals from P_t to T_t, but based on adapt_scores:
        the higher the score the greater propability to be copied (reproduced)

        return : new temporary population T_t
    """
    # logging.info("Reproducing...")
    adapt_scores = np.array(adapt_scores)
    logging.debug("adapt_scores: {}".format(adapt_scores))
    # probabilities to be reproduced
    prob_repr = (adapt_scores - adapt_scores.min()) / (np.sum(adapt_scores) - adapt_scores.min())
    # cummulative probability (normilized)
    prob_repr = np.cumsum(prob_repr / sum(prob_repr))
    logging.debug("cumulated prob_repr: {}".format(prob_repr))

    n = len(pop_t)
    new_indices = [np.argmax(np.random.random() < prob_repr) for i in range(n)]
    # new_indices = np.zeros(n)
    # for i in range(n):
    #     new_indices[i] = np.argmax(np.random.random() < prob_repr)
    logging.debug("new_indices: {}".format(new_indices))
    logging.debug("reproduction - end")

    return pop_t[new_indices]


def crossover(temp_pop_t, prob_cross = 0.7):
    """
        Simple one point crossover of chromosomes (in temporary population T_t).

        Steps:
        - split population into pairs
        - crossover with probability prob_cross, choose randomly the place to split with uniform distribution

        return : modified temporary population T_t
    """
    # logging.info("Crossover...")
    mod_pop = np.zeros(temp_pop_t.shape)
    n_pairs = len(temp_pop_t) // 2
    cut_bools = np.random.rand(n_pairs)
    cut_places = np.random.randint(1, len(temp_pop_t[0][0]), size=n_pairs)
    pairs = [i for i in range(n_pairs)]

    for pair_i, cut_bool, cut_i in zip(pairs, cut_bools, cut_places):
        if cut_bool:
            parent_1 = temp_pop_t[2*pair_i]
            parent_2 = temp_pop_t[2*pair_i + 1]
            mod_pop[2*pair_i] = np.hstack((parent_1[:, :cut_i], parent_2[:, cut_i:]))
            mod_pop[2*pair_i+1] = np.hstack((parent_2[:, :cut_i], parent_1[:, cut_i:]))

    logging.debug("crossover - end")
    return mod_pop


# TODO : check how mutation works
def mutation(pop_o_t, prob_mutation = 0.1, min=-1, max=1):
    """
        Mutation of each gene with probability prob_mutation.
        If mutate, choose new gene value from corresponding range between min and max

        return : new child population O_t
    """
    # logging.info("mutation - start")
    for i_i, ind in enumerate(pop_o_t):
        for c_i, centroid in enumerate(ind):
            mutate_bools = np.random.rand(len(centroid)) < prob_mutation
            mutate_vals = (max - min) * np.random.random(len(centroid)) + min
            centroid[mutate_bools] = mutate_vals[mutate_bools]
            pop_o_t[i_i, c_i, :] = centroid[:]
            logging.debug("indiv\t{}\tcentroid\t{}\tmutated genes:\t{} of {}".format(i_i, c_i, np.sum(mutate_bools), len(centroid)))
    logging.debug("mutation - end")
    return pop_o_t


def run_SGA(iter_num, data, labs, pop_num, prob_cross, prob_mutation, centers_num, 
            adapt_function, dist_measure, log_dir="logs"):
    """
        Run the whole Simple Genetic Algotithm.
        
        iter_num        : number of generations to calculate
        data            : data to carry on experiment
        labs            : true labels for data
        pop_num         : number of individuals in population
        prob_cross      : crossover probability
        prob_mutation   : mutation probability
        centers_num     : number of cluster to create
        adapt_function  : {"silhouette"|"info_gain"}; these values to be returned
        dist_measure    : {"euclidian"|"manhattan"|"cosine"} used for clustering and silhouette score
        log_dir         : directory for results output

        return : [tuple] (iterations [list], scores [list of lists], 
                        generations [list of individuals], total_time [in seconds])
    """
    # logging.info("run_SGA - start")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time()

    iterations, scores, generations = [], [], []

    pop, maxima, minima = init_population(data, centers_num, pop_num)
    pop_scores = get_adapt_scores(pop, data, labs, adapt_function=adapt_function, dist_measure=dist_measure)
    iterations.append(0)
    scores.append(pop_scores)
    generations.append(pop)

    for it in range(iter_num):
        logging.info("Generation: {}".format(it+1))
        pop = reproduction(pop, pop_scores)
        pop = crossover(pop, prob_cross)
        pop = mutation(pop, prob_mutation, min=minima, max=maxima)
        pop_scores = get_adapt_scores(pop, data, labs, adapt_function=adapt_function, dist_measure=dist_measure)
        iterations.append(it+1)
        scores.append(pop_scores)
        generations.append(pop)

    iterations = np.array(iterations)
    scores = np.array(scores)
    generations = np.array(generations)

    # test result
    total_time = time.time() - start_time
    logging.debug("run_SGA - end")

    logging.info("writing log")
    log_dir = os.path.join(log_dir, "_".join(
        [timestamp, str(pop_num), str(prob_cross), str(prob_mutation), str(centers_num), adapt_function, dist_measure]))
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, timestamp + ".txt"), "w") as out_f:
        out_f.write("evolution params:\n")
        out_f.write("iter_num:\t{}\n".format(iter_num))
        out_f.write("data shape:\t{}\n".format(data.shape))
        out_f.write("labs shape:\t{}\n".format(labs.shape))
        out_f.write("pop_num:\t{}\n".format(pop_num))
        out_f.write("prob_cross:\t{}\n".format(prob_cross))
        out_f.write("prob_mutation:\t{}\n".format(prob_mutation))
        out_f.write("centers_num:\t{}\n".format(centers_num))
        out_f.write("adapt_function:\t{}\n".format(adapt_function))
        out_f.write("dist_measure:\t{}\n".format(dist_measure))
        out_f.write("log_dir:\t{}\n".format(log_dir))
        out_f.write("----------------------------\n")
        out_f.write("results\n")
        out_f.write("best score:\t{}\n".format(scores.max()))
        out_f.write("best score (index):\t{}\n".format(scores.argmax()))
        out_f.write("total_time:\t{}\n".format(total_time))
    logging.info("writing log ended. time: {}".format(total_time))

    return iterations, scores, generations, total_time
