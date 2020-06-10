# Information-Theoretic Approximation to Causal Models

This repository is the official implementation of Information-Theoretic Approximation to Causal Models. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Usage

This package offers methods for causal discovery using information-theoretic approximation to causal models for both
i.i.d. data and timeseries data.

For i.i.d. data use iacm_discovery() and for timeseries data use iacm_discovery_timeseries()

To test these functions generate some sample data from an underlying model X->Y using the function generate_discrete_data():

```python
from iacm.data_generation import generate_discrete_data

data = generate_discrete_data(structure='nonlinear_discrete', sample_size=100, alphabet_size_x=4, alphabet_size_y=4)
```
Apply iacm_discovery using auto configured parameters
```python
res, criteria = iacm_discovery(base_x=2, base_y=2, data=data, auto_configuration=True,
                               parameters=None, verbose=False, preserve_order=False)

print("Ground Truth: X->Y")
print("IACM Causal Discovery using auto configuration: " + res + " (error: " + str(crit) + ")")
```

We can also approximate the sample data to a monotone model and calculate probabilities how necessary, sufficient, and necessary and sufficient a cause is for its effect (PN, PS, PNS).

```python
pn, ps, pns, error = get_causal_probabilities(data=data, auto_configuration=True, parameters=None, 
                                              direction_x_to_y=True, preserve_order=False)

print("Approximation to best monotone model X->Y gives error " + str(error) + " and ")
print("PN: " + str(pn))
print("PS: " + str(ps))
print("PNS: " + str(pns))
```

If ```auto_configuration=True``` the ```parameters``` in the functions above will be ignored and generated following a simple heuristic
described in the Supplementary Material of the paper. If ```auto_configuration=False```
parameters are expected to be set. Parameters is a dictionary consisting of the following entries:

```python
parameters = {'bins': 10,
              'nb_cluster': 2,
              'preprocess_method': 'cluster_discrete',
              'decision_criteria': 'global_error'}
``` 
The ```'bins'``` define the number of bins used during discretization in the preprocessing. The ```'nb_cluster'``` entry specifies the number of clusters
used at the clustering step in the preprocessing. The entry ```'preprocess_method'``` specifies the method used for preprocessing of the data. 
Currently there are the following possibilities:
 - ```'none'```: no preprocessing.
 - ```'cluster_discrete'```: cluster the data using KMeans and 'nb_cluster' and discretize data using KBinsDiscretizer and 'bins'.
 - ```'discrete_cluster'```: discretize data using 'bins' followed by a KMeans clustering using 'nb_cluster'.
 
The entry ```'decision_criteria'``` specifies which decision criteria is used in order to decide the causal direction. You can either use
```'global_error'``` or ```'kl_xy'``` as a decision criterion. See paper and Supplementary Material for more details.
