"""
Simple Genetic Algorithm
"""
import cluster
import numpy as np
import time
from datetime import datetime

import os
import logging

logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.DEBUG)

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
    # logging.debug("data maxima:\t{}".format(maxima))
    # logging.debug("data minima:\t{}".format(minima))
    # logging.debug("data maxima len:\t{}".format(len(maxima)))
    pop = (maxima - minima) * np.random.random_sample((n, centers_num, len(data[0]))) + minima
    logging.debug("init_population - end")
    return pop, maxima, minima


def get_adapt_scores(pop_t, data, true_labs, adapt_function="silhouette", dist_measure="euclidean", loggin_pref=""):
    """
        Cluster data for eacn individual (centroids set) in population and get scores.

        pop_t           : population to score
        true_labs       : for data provided
        adapt_function  : {"silhouette"|"info_gain"}; these values to be returned
        dist_measure    : {"euclidean"|"manhattan"|"cosine"} used for clustering and silhouette score

        return : scores for each individual; 1-D array of length n (pop_t size)
    """
    # logging.info("getting adapt scores...")
    # logging.debug("pop_t len:\t{}".format(len(pop_t)))
    # logging.debug("data len:\t{}".format(len(data)))
    # logging.debug("adapt_function:\t{}".format(adapt_function))
    # logging.debug("dist_measure:\t{}".format(dist_measure))

    true_num = len(np.unique(true_labs))
    # TODO: vectorize
    scores = []
    for i, individual in enumerate(pop_t):
        logging.debug("{}indiv {}: Clustering...".format(loggin_pref, i))

        labs = cluster.Centroids.cluster(data, individual, dist_func=dist_measure)
        uniq_num = len(np.unique(labs))
        logging.debug("{}indiv {}: labs : {}".format(loggin_pref, i, labs))

        logging.debug("{}indiv {}: labs unique num: {}".format(loggin_pref, i, uniq_num))
        if adapt_function == "silhouette":
            if uniq_num == 1:
                cur_score = -1
            elif uniq_num < true_num:
                logging.debug("{}indiv {}: computing silhouette...".format(loggin_pref, i))
                cur_score = cluster.Evaluate.silhouette(data, labs, dist_func=dist_measure)
                corrected_score = cur_score - (true_num - uniq_num) * 0.1
                cur_score = max(corrected_score, -1)
            else:
                logging.debug("{}indiv {}: computing silhouette...".format(loggin_pref, i))
                cur_score = cluster.Evaluate.silhouette(data, labs, dist_func=dist_measure)
            logging.debug("{}indiv {}: cur_score: {}".format(loggin_pref, i, cur_score))
            scores.append(cur_score)
        elif adapt_function == "info_gain":
            logging.debug("{}indiv {}: computing info_gain...".format(loggin_pref, i))
            labs = cluster.Utils.adjust_labels(labs, true_labs)
            logging.debug("{}indiv {}: adjust_labels: {}".format(loggin_pref, i, labs))
            logging.debug("{}indiv {}: adjust_labels length: {}".format(loggin_pref, i, len(labs)))
            cur_score = cluster.Evaluate.information_gain(true_labs, labs)
            logging.debug("{}indiv {}: cur_score: {}".format(loggin_pref, i, cur_score))
            scores.append(cur_score)
    
    # correction
    if adapt_function == "silhouette":
        uniq_scores = set(scores)
        # logging.debug("uniq_scores len {}".format(len(uniq_scores)))
        # logging.debug("uniq_scores {}".format(uniq_scores))
        second_min = sorted(uniq_scores)[1]
        scores = np.array(scores)
        scores[scores == -1] = second_min

    # logging.debug("scores: {}".format(scores))
    logging.debug("DONE - getting adapt scores")
    return scores


def reproduction(pop_t, adapt_scores, loggin_pref=""):
    """
        Randomly copy individuals from P_t to T_t, but based on adapt_scores:
        the higher the score the greater propability to be copied (reproduced)

        return : new temporary population T_t
    """
    # logging.info("Reproducing...")
    adapt_scores = np.array(adapt_scores)
    logging.debug("{}adapt_scores: {}".format(loggin_pref, adapt_scores))
    # probabilities to be reproduced
    prob_repr = (adapt_scores - adapt_scores.min()) / (np.sum(adapt_scores) - adapt_scores.min())
    # cummulative probability (normilized)
    prob_repr = np.cumsum(prob_repr / sum(prob_repr))
    logging.debug("{}cumulated prob_repr: {}".format(loggin_pref, prob_repr))

    n = len(pop_t)
    new_indices = [np.argmax(np.random.random() < prob_repr) for i in range(n)]
    # new_indices = np.zeros(n)
    # for i in range(n):
    #     new_indices[i] = np.argmax(np.random.random() < prob_repr)
    logging.debug("{}new_indices: {}".format(loggin_pref, new_indices))
    # logging.debug("reproduction - end")

    return pop_t[new_indices]


def crossover(temp_pop_t, prob_cross = 0.7, loggin_pref=""):
    """
        Simple one point crossover of chromosomes (in temporary population T_t).

        Steps:
        - split population into pairs
        - crossover with probability prob_cross, choose randomly the place to split with uniform distribution

        return : modified temporary population T_t
    """
    logging.info("{}Crossover...".format(loggin_pref))
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

    # logging.debug("crossover - end")
    return mod_pop


# TODO : check how mutation works
def mutation(pop_o_t, prob_mutation = 0.1, min=-1, max=1, loggin_pref=""):
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
            logging.debug("{}indiv:\t{};\tcentroid:\t{};\tmutated genes:\t{} of {}".format(loggin_pref, i_i, c_i, np.sum(mutate_bools), len(centroid)))
    logging.debug("mutation - end")
    return pop_o_t


def run_SGA(iter_num, data, labs, pop_num, prob_cross, prob_mutation, centers_num, 
            adapt_function, dist_measure, log_dir="logs", loggin_pref=""):
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
                        generations [list of individuals], total_time [in seconds]), log_dir
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
        logging.info("\n\n")
        logging.info("#################################")
        logging.info("### Generation: {:6}/{:6} ###".format(it+1, iter_num))
        logging.info("#################################\n")
        log_pref = "{}gen {}: ".format(loggin_pref, it+1)
        pop = reproduction(pop, pop_scores, loggin_pref=log_pref)
        pop = crossover(pop, prob_cross, loggin_pref=log_pref)
        pop = mutation(pop, prob_mutation, min=minima, max=maxima, loggin_pref=log_pref)
        pop_scores = get_adapt_scores(pop, data, labs, adapt_function=adapt_function, dist_measure=dist_measure, loggin_pref=log_pref)
        iterations.append(it+1)
        scores.append(pop_scores)
        generations.append(pop)

    iterations = np.array(iterations)
    scores = np.array(scores)
    generations = np.array(generations)

    # test result
    total_time = time.time() - start_time
    logging.debug("run_SGA - end")

    logging.info("writing log...")
    log_dir = os.path.join(log_dir, "_".join(
        [timestamp, "pop"+str(pop_num), "pc"+str(prob_cross), "pm"+str(prob_mutation), "centrs"+str(centers_num),
        "iters"+str(iter_num), adapt_function, dist_measure, "ds"+str(len(labs))]))
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
        out_f.write("total_time:\t{}min{}s\n".format(total_time//60, int(total_time%60)))
    logging.info("writing log ended. time: {}".format(total_time))
    logging.info("saving experiment output...")
    with open(os.path.join(log_dir, "iterations.npy"), "wb") as iters_f:
        np.save(iters_f, iterations)
    with open(os.path.join(log_dir, "generations.npy"), "wb") as gens_f:
        np.save(gens_f, generations)
    with open(os.path.join(log_dir, "scores.npy"), "wb") as scores_f:
        np.save(scores_f, scores)
    logging.info("saving experiment output... Done")

    return iterations, scores, generations, total_time, log_dir
