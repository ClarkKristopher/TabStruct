# TabStruct: Measuring Structural Fidelity of Tabular Data

![TabStruct banner](https://s2.loli.net/2025/05/16/TZ1clpvNBDhi8AE.png)

## 📌 Overview

**TabStruct** is an end‑to‑end benchmark for **tabular data generation, prediction, and evaluation**.  It ships with ready‑to‑use pipelines for

* generating high‑quality synthetic tables,
* training predictive models, and
* analysing results with a rich suite of metrics—especially those that quantify **structural fidelity**.

All components are designed to plug‑and‑play, so you can mix, match, and extend them to suit your own workflow.

```text
TabStruct/
├── src/               # Core library
│   ├── common/        # Shared utilities & configuration
│   ├── experiment/    # High‑level experiment runners
│   ├── generation/    # Synthetic‑data pipeline
│   └── prediction/    # Downstream‑task pipeline
├── private_repos/     # Patched third‑party repos
├── tests/             # Unit & integration tests
├── environment.yml    # Conda environment spec
├── pyproject.toml     # Project metadata & build config
└── README.md          # You are here
```

## 📚 Key Features

### Data generation

* Out‑of‑the‑box support for popular tabular generators: **SMOTE, TVAE, CTGAN, NFlow, TabDDPM, ARF**, and more.

### Evaluation metrics

* **Density estimation** – How well does the synthetic data approximate the real distribution?
* **Privacy preservation** – Does the generator leak sensitive records?
* **ML efficacy** – How do models trained on synthetic data perform compared to real data?
* **Structural fidelity** – Does the generator respect the causal structures of real data?

### Predictive tasks

* Classification & regression pipelines built on **scikit‑learn**, with optional neural‑network back‑ends.

## 🚀 Installation

We recommend managing dependencies with **conda** + **mamba**.

```bash
# 1️⃣ Upgrade conda and activate the base env
a conda update -n base -c conda-forge conda
conda activate base

# 2️⃣ Install the high‑performance dependency resolver
conda install conda-libmamba-solver --yes
conda config --set solver libmamba
conda install -c conda-forge mamba --yes

# 3️⃣ Create the project environment (≈5‑10 min)
mamba env create -f environment.yml
conda activate tabstruct

# 4️⃣ (Optional) re‑enable mamba in the new env
conda install conda-libmamba-solver --yes
conda config --set solver libmamba
conda install -c conda-forge mamba --yes

# 5️⃣ Install patched third‑party libraries
pip install -r private_repos/synthcity-private/requirements.txt
mamba install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install -e private_repos/synthcity-private[goggle]
pip install -e private_repos/Camel
pip install -e private_repos/tabpfn
```

> **Heads‑up:** Search the codebase for absolute paths and replace them with paths on your machine.

## 📊 Logging with W\&B

TabStruct logs every experiment to **Weights & Biases** (W\&B).  Use the default project or set your own credentials in `src/common/__init__.py`:

```python
WANDB_ENTITY  = "tabular-data-generation"
WANDB_PROJECT = "TabStruct"
```

## ✅ Quick sanity check

Run a toy classification job (K‑NN on the **Adult** dataset):

```bash
python -m src.experiment.run_experiment \
  --model knn \
  --save_model \
  --dataset adult \
  --test_size 0.2 \
  --valid_size 0.1 \
  --tags ENV-TEST
```

A successful run prints a series of **green** log lines like:

```text
[YYYY‑MM‑DD] Codebase: >>>>>>>>>> Launching create_data_module() <<<<<<<<<<<
…
```

If you see those, congratulations – your environment is ready! 🎉

## 💥 Example Workflows

### 1. Generate synthetic data

```bash
python -m src.experiment.run_experiment \
  --pipeline generation \
  --model smote \
  --generation_only \
  --dataset mfeat-fourier \
  --test_size 0.2 \
  --valid_size 0.1 \
  --tags SMOTE-generation
```

Template script: `scripts/template/generation/smote_generation.sh`.

---

### 2. Evaluate synthetic data

```bash
python -m src.experiment.run_experiment \
  --pipeline generation \
  --model smote \
  --eval_only \
  --dataset mfeat-fourier \
  --test_size 0.2 \
  --valid_size 0.1 \
  --generator_tags SMOTE-generation \
  --tags SMOTE-evaluation
```

Template script: `scripts/template/generation/smote_evaluation.sh`.

---

### 3. Launch a W\&B sweep (multiple data splits)

1. **Create the sweep config**
   Template: `scripts/template/sweep/baseline_generation_classification.yaml` (5 generators × 3 datasets × 10 splits).
2. **Initialise the sweep**

   ```bash
   wandb sweep scripts/template/sweep/baseline_generation_classification.yaml
   ```
   The command returns an agent ID (e.g. `tabular-data-generation/TabStruct/xxxxxxxx`).
3. **Launch agents**

   ```bash
   wandb agent AGENT_ID
   ```
   Each agent executes the full sweep defined above.

## 📀 Public Datasets Employed in TabStruct

* SCM datasets from bnlean

  * Classification

    ![image-20250515222136650](https://s2.loli.net/2025/05/16/eRDaPFAkiqwszxj.png)

  * Regression

    ![image-20250515222153930](https://s2.loli.net/2025/05/16/ivnoJ971cUzFyA3.png)

* Real-world datasets from OpenML and UCI

  * Classification

    ![image-20250515222104899](https://s2.loli.net/2025/05/16/dpzIo1fZqGQbBXy.png)

  * Regression

    ![image-20250515222120220](https://s2.loli.net/2025/05/16/JqVh3gFoxrKPck7.png)

