import sys
import time
import cluster
import data


def parse_args(arguments):

    dataset = "test"
    aggregate = True

    for args in arguments:

        if args.startswith("dataset"):
            dataset = args.split("=")[1]

        if args.startswith("aggregate"):
            if args.split("=")[1].lower() == "false":
                aggregate = False
            else:
                aggregate = True

    return dataset, aggregate


def get_samples_labels_datasets(dataset, aggregate):
    """
    return: [tuple of lists] (samples, labels, datasets)
    """
    d = data.HAPT()
    d.load_all_data()

    test_l = d.get_test_labels()
    train_l = d.get_train_labels()

    if aggregate:
        print("data aggregation...")
        d.aggregate_groups()
        test_l = d.get_aggregated_test_labels()
        train_l = d.get_aggregated_train_labels()

    test_s = d.get_test_data()
    train_s = d.get_train_data()

    if dataset == "all":
        return [test_s, train_s], [test_l, train_l], ["test_set", "train_set"]
    elif dataset == "train":
        return [train_s], [train_l], ["train_set"]
    return [test_s], [test_l], ["test_set"]


def test_silhouette(dataset = "test", aggregate = True):
    """
    Function to test scores and times of clustering for different distance
    measures available in scikit-learn on HAPT data. (Only chosen are tested)

    :param dataset: {"test" (default) / "train" / "all"}
    :param aggregate: {True (default) / False}
    :return:
    """

    print("=== test_silhouette started ===")
    total_test_time_start = time.time()

    dist_functions = ["cosine", "braycurtis", "canberra", "chebyshev", "correlation",
                      "hamming", "sqeuclidean", "euclidean", "manhattan"]

    # use test (default), train or both sets
    samples, labels, datasets = get_samples_labels_datasets(dataset, aggregate)

    scores = []
    print("{:13}".format("distance"), end="")
    for i in range(len(datasets)):
        print(" | {}:\t{}\t{}".format("data set", "score", "time"), end="")
    print()

    for dist_f in dist_functions:
        # cur_scores = []
        # cur_times = []
        print("{:13}".format(dist_f), end="")
        for i, (ds, labs, smpls) in enumerate(zip(datasets, labels, samples)):
            cur_start = time.time()
            s = cluster.Evaluate.silhouette(smpls, labs, dist_func=dist_f)
            t = time.time() - cur_start
            print(" | {}:\t{:.3f}\t{:.2f}".format(ds, s, t), end="")
            if i == len(datasets) - 1:
                print()

    print("total time of silhouette test: {}".format(time.time() - total_test_time_start))
    print("=== test_silhouette ended ===")


def main():
    dataset, aggregate = parse_args(sys.argv)
    test_silhouette(dataset, aggregate)


if __name__ == "__main__":

    main()
