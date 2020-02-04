## dl-kaggle-dataset-analysis

### Introduction

This project's goal is to analyse the dataset and apply classification of the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) with different models.
As we progress through the analysis, we will publish our results on this repository. Some of the results we will find might not be totally accurate.

Although we'll try to comment and precise every function and script in our files.

<img src="https://i.imgur.com/sWAAh1z.png">


<small><small>Description: the different classes of the CIFAR-10 dataset [here is the source](https://www.cs.toronto.edu/~kriz/cifar.html)</small></small>

### Quick Start

If this is the first time running this repository, launch:

```
python -m tools.create_new_environment
```

#### Jupyter notebooks

After that, you can directly look for the notebooks, by launching

```
cd notebooks
jupyter notebook
```


### Tools

#### Tensorboard

You can also directly look for the models generated on this repository to compare the results by launching :

```
python -m tools.tb
```

#### Requirements

If you want to refresh the `requirements.txt` file with your current Anaconda environment packages, you can launch :

```
python -m tools.refresh_requirements
```

#### Purge checkpoints

If you want to purge the checkpoints folder, you can launch:

```
python -m tools.purge_checkpoints
```

#### Purge model

If you want to purge a model , you can launch:

```
python -m tools.purge_model [model_name]
```

`[model_name]` in this case can be `mlp_100` for example.
So if we `launch python tools\purge_model.py mlp_100` it will delete all files related to the `mlp_100` model.


#### Authors

<a href="https://github.com/ElifCilingir">ElifCilingir</alt>
<br>
<a href="https://github.com/hakimMzabi">hakimMzabi</alt>
<br>
<a href="https://github.com/Mnadege">Mnadege</alt>
<br>
<a href="https://github.com/TheoHd">TheoHd</alt>