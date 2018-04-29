Proximal Backpropagation
================
Proximal Backpropagation (ProxProp) is a neural network training algorithm that takes *implicit* instead of *explicit* gradient steps to update the network parameters.
We have analyzed this algorithm in our ICLR 2018 paper:

**Proximal Backpropagation** (Thomas Frerix, Thomas Möllenhoff, Michael Moeller, Daniel Cremers; ICLR 2018) [https://arxiv.org/abs/1706.04638]

tl;dr
-------------------
- We provide a PyTorch implementation of ProxProp for Python 3 and PyTorch 0.4.0.
- The results of our paper can be reproduced by executing the script `paper_experiments.sh`.
- ProxProp is implemented as a `torch.nn.Module` (a 'layer') and can be combined with any other layer and first-order optimizer.
While a ProxPropConv2d and a ProxPropLinear layer already exist, you can generate a ProxProp layer for your favorite linear layer with one line of code.

Installation
-------------------
1. Make sure you have a running Python 3 (>=3.5) ecosytem. We recommend that you use a [conda](https://conda.io/docs/) install, as this is also the recommended option to get the latest PyTorch running. 
For this README and for the scripts, we assume that you have `conda` running with Python 3.5.
2. Clone this repository and switch to the directory.
3. Install the dependencies via `conda install --file conda_requirements.txt` and `pip install -r pip_requirements.txt`.
4. Install [PyTorch](http://pytorch.org/) with magma support. 
    We have tested our code with PyTorch 0.4.0 and CUDA 9.0.
    You can install this setup via
    ```
    conda install -c pytorch magma-cuda90
    conda install pytorch cuda90 -c pytorch
    ```
5. (optional, but necessary to reproduce paper experiments) Download the CIFAR-10 dataset by executing `get_data.sh`

Training neural networks with ProxProp
-------------------
ProxProp is implemented as a custom linear layer (`torch.nn.Module`) with its own backward pass to take implicit gradient steps on the network parameters. 
With this design choice it can be combined with any other layer, for which one takes explicit gradient steps. 
Furthermore, the resulting update direction can be used with any first-order optimizer that expects a suitable update direction in parameter space.
In our [paper](https://arxiv.org/abs/1706.04638) we prove that ProxProp generates a descent direction and show experiments with Nesterov SGD and Adam.

You can use our pre-defined layers `ProxPropConv2d` and `ProxPropLinear`, corresponding to `nn.Conv2d` and `nn.Linear`, by importing

`from ProxProp import ProxPropConv2d, ProxPropLinear`

Besides the usual layer parameters, as detailed in the [PyTorch docs](http://pytorch.org/docs/master/), you can provide:

- `tau_prox`: step size for a proximal step; default is `tau_prox=1`
- `optimization_mode`: can be one of `'prox_exact'`, `'prox_cg{N}'`, `'gradient'` for an exact proximal step, an approximate proximal step with `N` conjugate gradient steps and an explicit gradient step, respectively; default is `optimization_mode='prox_cg1'`.
The `'gradient'` mode is for a fair comparison with SGD, as it incurs the same overhead as the other methods in exploiting a generic implementation with the provided PyTorch API. 

If you want to use ProxProp to optimize your favorite linear layer, you can generate the respective module with one line of code.
As an example for the the `Conv3d` layer:

```
from ProxProp import proxprop_module_generator
ProxPropConv3d = proxprop_module_generator(torch.nn.Conv3d)
```

This gives you a default implementation for the approximate conjugate gradient solver, which treats all parameters as a stacked vector.
If you want to use the exact solver or want to use the conjugate gradient solver more efficiently, you have to provide the respective reshaping methods to `proxprop_module_generator`, as this requires specific knowledge of the layer's structure and cannot be implemented generically.
As a template, take a look at the `ProxProp.py` file, where we have done this for the `ProxPropLinear` layer.

By reusing the forward/backward implementations of existing PyTorch modules, ProxProp becomes readily accessible.
However, we pay an overhead associated with generically constructing the backward pass using the PyTorch API.
We have intentionally sided with genericity over speed.

Reproduce paper experiments
-------------------
To reproduce the paper experiments execute the script `paper_experiments.sh`.
This will run our paper's experiments, store the results in the directory `paper_experiments/` and subsequently compile the results into the file `paper_plots.pdf`.
We use an NVIDIA Titan X GPU; executing the script takes roughly 3 hours.

Acknowledgement
-------------------
We want to thank [Soumith Chintala](https://github.com/soumith) for helping us track down a mysterious bug and the whole PyTorch dev team for their continued development effort and great support to the community.

Publication
-------------------
If you use ProxProp, please acknowledge our paper by citing

```
@article{Frerix-et-al-18,
    title = {Proximal Backpropagation},
    author={Thomas Frerix, Thomas Möllenhoff, Michael Moeller, Daniel Cremers},
    journal={International Conference on Learning Representations},
    year={2018},
    url = {https://arxiv.org/abs/1706.04638}
}
```
