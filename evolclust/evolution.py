"""
Simple Genetic Algorithm
"""
import cluster
import numpy as np
import time
from datetime import datetime

import os
import logging

LOG_LEVEL =  "info" # "debug" # "info"
if LOG_LEVEL == "debug":
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.DEBUG)
else:
    logging.basicConfig(format='%(message)s', level=logging.INFO)


def init_population(data, centers_num=12, n=20):
    """
        Choose randomly genes for chromosomes between minima and maxima of 
        corresponding genes in data

        data        : data to get minima and maxima for generation
        centers_num : number of centroids per individual to generate
        n           : number of indiwiduals

        return : initial population pop_0, maxima and minima for mutation
    """
    logging.info("initialize population...")
    data = np.array(data)
    maxima = data.max(axis=0)
    minima = data.min(axis=0)
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
    logging.debug("{}getting adapt scores...".format(loggin_pref))

    true_num = len(np.unique(true_labs))
    # TODO: vectorize
    scores = []
    for i, individual in enumerate(pop_t):
        logging.debug("{}indiv {}: Clustering...".format(loggin_pref, i))
        labs = cluster.Centroids.cluster(data, individual, dist_func=dist_measure)

        uniq_num = len(np.unique(labs))
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
            logging.debug("{}indiv {}: adjust_labels length: {}".format(loggin_pref, i, len(labs)))
            cur_score = cluster.Evaluate.information_gain(true_labs, labs)
            logging.debug("{}indiv {}: cur_score: {}".format(loggin_pref, i, cur_score))
            scores.append(cur_score)
    
    # correction (-1 replace with second minimal value) & normalization
    if adapt_function == "silhouette" and len(pop_t) > 1:
        scores = np.array(scores)
        scores[scores == -1] = sorted(set(scores))[1]
        scores = (scores + 1)/2

    # logging.debug("scores: {}".format(scores))
    logging.debug("{}getting adapt scores... DONE".format(loggin_pref))
    return scores


def reproduction(pop_t, adapt_scores, loggin_pref=""):
    """
        Randomly copy individuals from P_t to T_t, but based on adapt_scores:
        the higher the score the greater propability to be copied (reproduced)

        return : new temporary population T_t
    """
    logging.debug("{}reproducing...".format(loggin_pref))
    adapt_scores = np.array(adapt_scores)
    logging.debug("{}adapt_scores len: {}".format(loggin_pref, len(adapt_scores)))
    # probabilities to be reproduced
    prob_repr = (adapt_scores - adapt_scores.min()) / (np.sum(adapt_scores) - adapt_scores.min())
    # cummulative probability (normilized)
    prob_repr = np.cumsum(prob_repr / sum(prob_repr))
    logging.debug("{}cumulative prob_repr len: {}".format(loggin_pref, len(prob_repr)))

    n = len(pop_t)
    new_indices = [np.argmax(np.random.random() < prob_repr) for i in range(n)]
    logging.debug("{}new_indices: {}".format(loggin_pref, new_indices))
    logging.debug("{}reproducing... DONE".format(loggin_pref))

    return pop_t[new_indices]


def crossover(temp_pop_t, prob_cross = 0.7, loggin_pref=""):
    """
        Simple one point crossover of chromosomes (in temporary population T_t).

        Steps:
        - split population into pairs
        - crossover with probability prob_cross, choose randomly the place to split with uniform distribution

        return : modified temporary population T_t
    """
    logging.debug("{}crossover...".format(loggin_pref))
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

    logging.debug("{}crossover... DONE".format(loggin_pref))
    return mod_pop


def mutation(pop_o_t, prob_mutation = 0.1, min=-1, max=1, loggin_pref=""):
    """
        Mutation of each gene with probability prob_mutation.
        If mutate, choose new gene value from corresponding range between min and max

        return : new child population O_t
    """
    logging.debug("{}mutation...".format(loggin_pref))
    for i_i, ind in enumerate(pop_o_t):
        for c_i, centroid in enumerate(ind):
            mutate_bools = np.random.rand(len(centroid)) < prob_mutation
            mutate_vals = (max - min) * np.random.random(len(centroid)) + min
            centroid[mutate_bools] = mutate_vals[mutate_bools]
            pop_o_t[i_i, c_i, :] = centroid[:]
            logging.debug("{}indiv {}; centroid {}; mutated genes {}/{}".format(loggin_pref, i_i, c_i, np.sum(mutate_bools), len(centroid)))
    logging.debug("{}mutation... DONE".format(loggin_pref))
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
                        generations [list of individuals], total_time [in seconds]), log_dir,
                        list of lists max,min,avg,median scores, tuple with indices of the best individual 
    """
    logging.info("{}Simple Genetic Algotithm Run".format(loggin_pref))
    logging.info("{}============================".format(loggin_pref))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time()

    iterations, scores, generations = [], [], []

    pop, maxima, minima = init_population(data, centers_num, pop_num)
    pop_scores = get_adapt_scores(pop, data, labs, adapt_function=adapt_function, dist_measure=dist_measure)
    iterations.append(0)
    scores.append(pop_scores)
    generations.append(pop)
    pop_scores = np.array(pop_scores)

    logging.info("{}generation:\tmax  \tmin  \tmean \tmedian".format(loggin_pref))
    logging.info("{}gen {}/{}:\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
        loggin_pref, 0, iter_num, pop_scores.max(), pop_scores.min(), pop_scores.mean(), np.median(pop_scores)))

    for it in range(iter_num):
        log_pref = "{}gen {}/{}: ".format(loggin_pref, it+1, iter_num)
        pop = reproduction(pop, pop_scores, loggin_pref=log_pref)
        pop = crossover(pop, prob_cross, loggin_pref=log_pref)
        pop = mutation(pop, prob_mutation, min=minima, max=maxima, loggin_pref=log_pref)
        pop_scores = get_adapt_scores(pop, data, labs, adapt_function=adapt_function, dist_measure=dist_measure, loggin_pref=log_pref)
        iterations.append(it+1)
        scores.append(pop_scores)
        generations.append(pop)
        pop_scores = np.array(pop_scores)

        logging.info("{}gen {}/{}:\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(loggin_pref, it+1, iter_num,
                     pop_scores.max(), pop_scores.min(), pop_scores.mean(), np.median(pop_scores)))

    iterations = np.array(iterations)
    scores = np.array(scores)
    generations = np.array(generations)

    # test result
    total_time = time.time() - start_time
    logging.debug("{}SGA RUN - DONE".format(loggin_pref))

    logging.info("{}writing log...".format(loggin_pref))
    log_dir = os.path.join(log_dir, "_".join(
        [timestamp, "pop"+str(pop_num), "pc"+str(prob_cross), "pm"+str(prob_mutation), "centrs"+str(centers_num),
        "iters"+str(iter_num), adapt_function, dist_measure, "ds"+str(len(labs))]))
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, timestamp + ".txt"), "w") as out_f:
        out_f.write("params:\n")
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
        out_f.write("-----------------------------------------------\n")
        out_f.write("results\n")
        out_f.write("best score:\t{}\n".format(scores.max()))
        best_generation_ind = scores.argmax() // scores.shape[1]
        best_ind_inbest_gen = scores.argmax() - (best_generation_ind * scores.shape[1])
        out_f.write("best score (index):\tgeneration {}, individual {}\n".format(best_generation_ind, best_ind_inbest_gen))
        out_f.write("total_time:\t{}min{}s\n".format(total_time//60, str(total_time%60)[:6]))
    logging.info("{}writing log... DONE. time: {}".format(loggin_pref, total_time))
    logging.info("{}saving experiment output...".format(loggin_pref))
    with open(os.path.join(log_dir, "iterations.npy"), "wb") as iters_f:
        np.save(iters_f, iterations)
    with open(os.path.join(log_dir, "generations.npy"), "wb") as gens_f:
        np.save(gens_f, generations)
    with open(os.path.join(log_dir, "scores.npy"), "wb") as scores_f:
        np.save(scores_f, scores)
    logging.info("{}saving experiment output... DONE".format(loggin_pref))

    return iterations, scores, generations, total_time, log_dir, (best_generation_ind, best_ind_inbest_gen)
