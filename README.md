# TabFairGAN

This repository is the code for the papar [**TabFairGAN: Fair Tabular Data Generation with Generative Adversarial Networks**](https://arxiv.org/abs/2109.00666) . TabFairGAN is a synthetic tabular data generator which could produce synthetic data, with or without _**fairness**_ constraint. The model uses a Wasserstein Generative Adversarial Network to produce synthetic data with high quality.


# Usage

TabFairGAN is used programmatically in Python. You can either generate synthetic data with fairness constraints or without fairness constraints.
The package now provides a more modular interface. 

## Basic Usage
1. Without Fairness Constraints: If you do not need fairness constraints, you simply omit the fairness_config parameter.
2. With Fairness Constraints: To enforce fairness constraints, you must pass a dictionary with specific parameters as explained below.

## Example 1: Without Fairness Constraints

```python
import pandas as pd
from tabfairgan import TFG

# Load your dataset
df = pd.read_csv("adult.csv")

# Initialize TabFairGAN without fairness constraints
tfg = TFG(df, epochs=200, batch_size=256, device='cuda:0')

# Train the model
tfg.train()

# Generate synthetic data
fake_df = tfg.generate_fake_df(num_rows=32561)

```


In this case, the model will focus solely on generating high-quality synthetic data without considering fairness.

## Example 2: With Fairness Constraints

To generate fair synthetic data, you need to pass a dictionary containing the following parameters:

* fair_epochs: Number of fair training epochs (integer).
* lamda: Lambda parameter controlling the trade-off between fairness and accuracy (float).
* S: Protected attribute (string, e.g., "sex").
* Y: Decision label (string, e.g., "income").
* S_under: Value representing the underprivileged group for the protected attribute (string, e.g., " Female").
* Y_desire: Desired value for the label (string, e.g., " >50K").



```python
import pandas as pd
from tabfairgan import TFG

# Load your dataset
df = pd.read_csv("adult/adult.csv")

# Define fairness configuration
fairness_config = {
    'fair_epochs': 50,
    'lamda': 0.5,
    'S': 'sex',
    'Y': 'income',
    'S_under': ' Female',
    'Y_desire': ' >50K'
}

# Initialize TabFairGAN with fairness constraints
tfg = TFG(df, epochs=200, batch_size=256, device='cuda:0', fairness_config=fairness_config)

# Train the model
tfg.train()

# Generate synthetic data
fake_df = tfg.generate_fake_df(num_rows=32561)
```

In this case, the model will generate synthetic data that not only preserves high quality but also enforces fairness with respect to the specified protected attribute and decision label.

## Important Notes:
* Fairness Configuration: If you want to use fairness constraints, you must provide a dictionary containing all the required fairness parameters: fair_epochs, lamda, S, Y, S_under, and Y_desire.
* Without Fairness: If no fairness_config is provided, the model will default to generating synthetic data without fairness constraints.



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


