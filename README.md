# TabFairGAN

This repository is the code for the papar [**TabFairGAN: Fair Tabular Data Generation with Generative Adversarial Networks**](https://arxiv.org/abs/2109.00666) . TabFairGAN is a synthetic tabular data generator which could produce synthetic data, with or without _**fairness**_ constraint. The model uses a Wasserstein Generative Adversarial Network to produce synthetic data with high quality.
___
# Prerequisites
- Python 3.7+
- [PyTorch](https://pytorch.org/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
___

# Usage

This repository currently only includes the cli of the proposed model. Here I explain how to use the model using an example: Adult Income Dataset. The csv file could be found in "adult" folder. The model could be used either with no fairness constraint, i.e. you only care about producing a high quality fake dataset and do not care about fairness, or you want to produce a high quality dataset which is also fair with respect to a binary protected attribute and binary decision (label).

The first argument given to the program is either ```with_fairness``` or ```no_fairness``` :
```
$ python TabFairGAN.py --help
usage: TabFairGAN.py [-h] {with_fairness,no_fairness} ...

positional arguments:
  {with_fairness,no_fairness}
```

## 1 - No Fairness
Below shows the parameters you need to specify for data generation:

```
$ python TabFairGAN.py no_fairness --help
usage: TabFairGAN.py no_fairness [-h] df_name num_epochs batch_size fake_name size_of_fake_data

positional arguments:
  df_name            Reference dataframe
  num_epochs         Total number of epochs
  batch_size         the batch size
  fake_name          name of the produced csv file
  size_of_fake_data  how many data records to generate
```

Example:

```
$python TabFairGAN.py no_fairness adult.csv 300 256 fake_adult.csv 32561

```
Where original dataset name is adult.csv, the model is trained for 300 epochs, the batchsize is 256, the produced fake data name would be fake_adult.csv and will contain 32561 records (rows).


## 2 - With Fairness

To produce a fair fake data, other parameters must be specified. You should specify the protected attribute, the underproviledged value for protected attribute, Labels, and the desirable value for label. Please note that **the protected attribute and the label must be binary**. The required parameters include:

```
$ python TabFairGAN.py with_fairness --help
usage: TabFairGAN.py with_fairness [-h]
                                   df_name S Y underprivileged_value desirable_value num_epochs batch_size num_fair_epochs lambda_val
                                   fake_name size_of_fake_data

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
  size_of_fake_data     how many data records to generate

```

For example for the case of Adult Income dataset, the data is shown to be biased against _" Female"_ gender. Therefore, the protected attribute is _"sex"_ (name of column in data), and the underpriviledged group value is _" Female"_. For the label (decision), the label name is _"income"_, and the desirable value for label is _" >50K"_. Here is an example:

```
$ python TabFairGAN.py with_fairness adult.csv "sex" "income" " Female" " >50K" 200 256 30 0.5 fake_adult.csv 32561

```
Produces a fake data with original data specified as adult.csv, protected attribute as _"income"_, underprivileged value for protected attribute as _" Female"_, label as _"income"_, label desirable value as _" 50K"_, 200 total epochs, batchsize of 256, 30 fair epochs, and a <img src="https://github.com/amirarsalan90/TabFairGAN/blob/main/lambda_f.png?raw=true" align="center" border="0" alt="\lambda_f" width="19" height="21" /> value of 0.5 ( <img src="https://github.com/amirarsalan90/TabFairGAN/blob/main/lambda_f.png?raw=true" align="center" border="0" alt="\lambda_f" width="19" height="21" /> is decsribed in the paper ). 


# Citing TabFairGAN

If you use TabFairGAN, please cite the paper:


```
@article{rajabi2022tabfairgan,
  title={Tabfairgan: Fair tabular data generation with generative adversarial networks},
  author={Rajabi, Amirarsalan and Garibay, Ozlem Ozmen},
  journal={Machine Learning and Knowledge Extraction},
  volume={4},
  number={2},
  pages={488--501},
  year={2022},
  publisher={MDPI}
}
```


