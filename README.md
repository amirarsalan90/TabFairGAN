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

This repository currently only includes the cli of the proposed model. Here I explain how to use the model using an example: Adult Income Dataset. The csv file could be found in "adult" folder. The model could be used either with no fairness constraint, i.e. you only care about producing a high quality fake dataset and do not care about fairness, or you want to produce a high quality dataset which is also fair with respect to a binary protected attribute and binary decision (label).

## 1 - No Fairness
Below shows the parameters you need to specify for data generation:

```
$ python TabFairGAN_nofair.py --help
usage: TabFairGAN_nofair.py [-h]
                            df_name num_epochs batch_size fake_name
                            size_of_fake_data

positional arguments:
  df_name            Reference dataframe
  num_epochs         Total number of epochs
  batch_size         the batch size
  fake_name          name of the produced csv file
  size_of_fake_data  how many data records to generate

optional arguments:
  -h, --help         show this help message and exit
```

The csv file of the original data should be in the same folder as your ```TabFairGAN_nofair.py``` file. For example:

```
$python TabFairGAN_nofair.py adult.csv 300 256 fake_adult.csv 32561

```
Where original dataset name is adult.csv, the model is trained for 300 epochs, the batchsize is 256, the produced fake data name would be fake_adult.csv and will contain 32561 records (rows).


## 2 - With Fairness

To produce a fair fake data, other parameters must be specified. For example, the required parameters include:

```
$ python TabFairGAN.py --help
usage: TabFairGAN.py [-h]
                     df_name S Y underprivileged_value desirable_value
                     num_epochs batch_size num_fair_epochs lambda_val
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

optional arguments:
  -h, --help            show this help message and exit

```
To produce a fair dataset, you should specify the protected attribute, the underproviledged value for protected attribute, Labels, and the desirable value for label. Please note that **the protected attribute and the label must be binary**. 
For the case of Adult Income dataset, the data is shown to be biased against " Female" gender. Therefore, the protected attribute is "sex" (name of column in data), and the underpriviledged group value is " Female". For the label (decision), the label name is "income", and the desirable value for label is " >50K". For example:









