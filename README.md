# evolclust - clustering-wtih-ewolution-strategy

Package for testing idea of using evolution strategies in clustering.

## Running

Running in `virtualenv` is recommended.

Requirements (versions used recently for runnin and testing):
- python 3.6
- pytest==4.1.0
- matplotlib==3.0.2
- numpy==1.15.4
- scipy==1.2.0
- scikit-learn==0.20.2

## Running tests

```
cd evolclust
python -m pytest .
```

## Silhouette test

Used to test Silhouette score depending on distance measure used.

Example use:

```
python3 silhouette_experiments.py dataset=train aggregate=false
```

> `dataset` = { test | train | all } <br>
> `aggregate` = { True | False }
