# Camel: A DataFrame-focused Solution for Tabular Dataset

## Prepare the environment

```bash
# Create a new conda environment

conda create -n tabular-data-test python=3.10
conda activate tabular-data-test

# Make conda faster

conda install -n base conda-libmamba-solver
conda config --set solver libmamba

# Install mamba, which is faster and ully compatible with conda commands

conda install -c conda-forge mamba

# Install AutoGluon

mamba install -c conda-forge autogluon "pytorch=_=cuda_"

# Install ray for faster training

mamba install -c conda-forge "ray-tune >=2.6.3,<2.7" "ray-default >=2.6.3,<2.7"
```
