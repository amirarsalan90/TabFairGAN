# TabFairGAN

This repository is the code for the papar in *** . TabFairGAN is a synthetic tabular data generator which could produce synthetic data, with or without _**fairness**_ constraint. The model uses a Wasserstein Generative Adversarial Network to produce fake data with high quality.
___
# Prerequisites
- Python 3.7+
- [PyTorch](https://pytorch.org/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
___

# Usage

This repository currently only includes the cli of the proposed model. Here I explain how to use the model using an example: Adult Income Dataset. The csv file could be found in "adult" folder. First, program parameters are as follows:

```
$ python TabFairGAN.py --help
usage: TabFairGAN.py [-h]
                     df_name S Y underprivileged_value desirable_value
                     num_epochs batch_size num_fair_epochs lambda_val
                     fake_name

positional arguments:
  df_name               Reference dataframe
  S                     Protected attribute
  Y                     Label (decision)
  underprivileged_value
                        Value for underpriviledged group
  desirable_value       Desired label (decision)
  num_epochs            Total number of epochs
  batch_size            the batch size
  num_fair_epochs       number of fair training epochs
  lambda_val            lambda parameter
  fake_name             name of the produced csv file

optional arguments:
  -h, --help            show this help message and exit

```


If you only want to produce a fake data from a reference dataset, and don't care about the fairness, you can do the following:






