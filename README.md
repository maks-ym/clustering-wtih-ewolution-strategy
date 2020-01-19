# evolclust - clustering-with-evolution-strategy

Package for testing idea of using evolution strategies in clustering.

- clustering slgorithm: `k-centroids`
- evolution algorithm: `classic`
- distance metrics: `cosine`, `euclidean`, `manhattan`
- evaluation functions: `silhouette`, `information gain`
- dataset: `HAPT`

## Requirements

- python 3.6
- pytest==4.1.0
- matplotlib==3.0.2
- numpy==1.15.4
- scipy==1.2.0
- scikit-learn==0.20.2

> Running in `virtualenv` is recommended.

## Running

### Evolution

**Example**:

```bash
(.venv) $ python3 main.py --iter_num=60 --pop_num=50 --prob_croevolclust
ss=0.5 --prob_mutation=0.02 --adapt_function=silh --dist_measure=cos --ag
gregate --data=test
```

**Parameters**:

```bash
'--iter_numn'       - umber of iterations
'--pop_num'         - size of population (min 2)
'--prob_cross'      - crossover probability
'--prob_mutation'   - mutation probability
'--aggregate'       - aggregate data groups
'--adapt_function'  - choices=['silh', 'info', 'info_gain', "silhouette"], silhouette or information gain
'--dist_measure'    - choices=['eucl', 'manh', 'cos', "euclidean", "manhattan", "cosine"] (for "euclidean", "manhattan", "cosine")
'--repeat'          - repeat experiment n times and average results
'--logdir'          - aggregate data groups
'--data'            - choices=['train', 'test'], aggregate data groups
'--showdata'        - only show data to be used in experiment
```

### Test of gotten solution

**Example**:

```bash
$ python3 inference.py --path=/home/invictus/Projects/evolclust/evolclust/logs/done/4_exp4_population_size/20190125_112245_pop200_pc0.01_pm0.01_centrs3_iters120_silhouette_cosine_ds3162
```

**Parameters**:

```bash
'--path PATH'       - directory with solution 'iterations.npy', 'scores.npy', 'generations.npy' and text file with experimet parameters
'--outdir OUTDIR'   - dir to output result plots (optional, by default put into "inference_output" in `PATH`)
```

## Running tests

### All at once

```bash
$ cd evolclust
$ python -m pytest .
```

### Chosen

Test function `test_euclidean_distance` from class `TestDistance` in file `test_cluster.py`:

```bash
$ cd evolclust
$ python -m pytest test/test_cluster.py::TestDistance::test_euclidean_distance
```

## Silhouette test

Used to test Silhouette score depending on distance measure used.
Outputs score and time. HAPT dataset is used.

**Example**:

```bash
$ python3 silhouette_experiments.py dataset=train aggregate=false
```

**Parameters**:

```bash
`dataset`   - which dataset to test. Options: test / train / all
`aggregate` - whether aggregate clusters. Options: True / False
```

## Folder structure

<pre>
evoclust -+
          +- __init__.py
          +- cluster.py
          +- data.py
          +- evolution.py
          +- main.py
          +- silhouette_experiments.py
          +- utils.py
          +- data -+
          |        +- hapt -+
          |                 +- activity_labels.txt
          |                 +- features.txt
          |                 +- features_info.txt
          |                 +- README.txt
          |                 +- raw_data -+
          |                 |            +- acc_exp01_user01.txt
          |                 |            :
          |                 |            +- acc_exp61_user30.txt
          |                 |            +- gyro_exp01_user01.txt
          |                 |            :
          |                 |            +- gyro_exp61_user30.txt
          |                 |            +- labels.txt
          |                 +- test -+
          |                 |        +- subject_id_rules.txt
          |                 |        +- x_test.txt
          |                 |        +- y_test.txt
          |                 +- train -+
          |                           +- subject_id_train.txt
          |                           +- x_train.txt
          |                           +- y_train.txt
          +- test -+
                   +- __init__.py
                   +- test_cluster.py
                   +- test_data.py
                   +- test_data -+
                                 +- subject_id_test.txt
                                 +- x_test.txt
                                 +- x_train.txt
                                 +- y_test.txt
                                 +- y_train.txt
</pre>
