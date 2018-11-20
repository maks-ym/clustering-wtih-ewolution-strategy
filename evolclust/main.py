#!/usr/bin/env python3

import data
import cluster


if __name__ == "__main__":
    # read data
    # convert (FFT)
    # train 
    # predict (test)
    # compare with k-means
    # visualize

    #read test
    file_path = "data/hapt/train/X_train.txt"
    train_data = []
    with open(file_path) as in_file:
        for line in in_file:
            parsed_line = [float(x) for x in line.rstrip("\n").split(" ")]
            train_data.append(parsed_line)
    
    for row in train_data:
        print(len(row))
    
    print(len(train_data))
