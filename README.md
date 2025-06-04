# VAEase Project

This repository contains code for experiments with various autoencoder models and synthetic datasets, as described in the associated paper.

## File Overview

- **models.py**  
  Contains the implementation of all model architectures used in the experiments, including encoders, decoders, and other neural network modules.

- **sync_dataset.py**  
  Implements the synthetic dataset classes described in the paper. This includes various data generation methods and dataset wrappers for training and evaluation.

- **Other Python Files**  
  All other `.py` files are runnable scripts. Their filenames follow the pattern:  
  ```
  dataset_modelmethod.py
  ```
  where `dataset` specifies the dataset used, and `modelmethod` specifies the model or method applied. For example, `activation_sae.py` runs the Sparse Autoencoder (SAE) on the activation dataset.

## Example Structure

- `activation_sae.py`: Runs SAE on the activation dataset.
- `embedding_vae.py`: Runs VAE on the embedding dataset.
- `fmnist_vaep.py`: Runs VAEP on the FashionMNIST dataset.

## Usage

To run an experiment, execute the corresponding Python script.  
For example:
```sh
python activation_sae.py
```

---

For more details on the datasets and models, please refer to the comments in [`models.py`](models.py) and [`sync_dataset.py`](sync_dataset.py).
