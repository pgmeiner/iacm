# Information-Theoretic Approximation to Causal Models

This repository is the official implementation of Information-Theoretic Approximation to Causal Models (https://arxiv.org/abs/2007.15047). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Usage

This package offers methods for causal discovery and the calculation of causal effects using information-theoretic approximation to causal models for both
i.i.d. data and timeseries data.

For i.i.d. data use iacm_discovery(), iacm_discovery_pairwise(), and for timeseries data use iacm_discovery_timeseries()

To test these functions generate some sample data from an underlying model X->Y using the function generate_discrete_data():

```python
from iacm.data_generation import generate_discrete_data

data = generate_discrete_data(structure='nonlinear_discrete', sample_size=100, alphabet_size_x=4, alphabet_size_y=4)
```
Apply iacm_discovery_pairwise using default parameters
```python
from iacm.iacm import iacm_discovery_pairwise

parameters = {'preprocess_method': 'none',
              'decision_criteria': 'global_error'}
result, error = iacm_discovery_pairwise(base_x=2, base_y=2, data=data, parameters=parameters, verbose=False, preserve_order=False)

print("Ground Truth: X->Y")
print("IACM Causal Discovery using auto configuration: " + result + " (error: " + str(error) + ")")
```

We can also approximate the sample data to a monotone model and calculate probabilities how necessary, sufficient, and necessary and sufficient a cause is for its effect (PN, PS, PNS).

```python
from iacm.iacm import get_causal_probabilities

pn, ps, pns, error = get_causal_probabilities(data=data, parameters=parameters, direction_x_to_y=True)

print("Approximation to best monotone model X->Y gives error " + str(error) + " and ")
print("PN: " + str(pn))
print("PS: " + str(ps))
print("PNS: " + str(pns))
```

We can also approximate the data to other causal models like X<-Z->Y. The model with the lowest error will be returned.
```python
from iacm.iacm import iacm_discovery

result, error = iacm_discovery(bases={'x': 2, 'y': 2}, data=data, parameters=parameters, causal_models=['X->Y', 'X<-Z->Y', 'X|Y'])
```

Parameters is a dictionary consisting of the following entries:

```python
parameters = {'preprocess_method': 'none',
              'decision_criteria': 'global_error'}
``` 
The entry ```'preprocess_method'``` specifies the method used for preprocessing of the data. 
Currently there are the following possibilities:
 - ```'none'```: no preprocessing, split data at half into observational and interventional data.
 - ```'split'```: filter data by each intervention on X or Y and draw without replacement from each filtered subset
            data points for observational and interventional data to obtain equally sized sets for observations and
            interventions with similar variance.
 - ```'split_and_balance'```: works like ```'split'``` but draws with replacement such that each intervention subset has
            equal size. This method tries to balance out imbalanced intervention subsets.
 
 
The entry ```'decision_criteria'``` specifies which decision criteria is used in order to decide the causal direction. You can either use
```'global_error'``` or ```'local_xy_error'``` as a decision criterion. See paper and Supplementary Material for more details.

You can find a running example in ```test/main.py```.

## Citing 
Please cite us when you are using this package as follows:

Peter Gmeiner: Information-Theoretic Approximation to Causal Models. 2020. https://arxiv.org/abs/2007.15047

Bibtex:
```bibtex
@article{iacm,
authors={Gmeiner, Peter},
title={Information-Theoretic Approximation to Causal Models},
howpublished={https://arxiv.org/abs/2007.15047}
year={2020}
}
```