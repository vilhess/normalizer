# Do Normalization Choice Matter for Causal Time-Series Foundation Models?

This repository contains the official code for the paper:

**Do Normalization Choice Matter for Causal Time-Series Foundation Models?**

---

## Repository Structure

- **`results/`**  
  Contains the experimental results reported in the paper, organized by experiment in separate subfolders.

- **`notebooks/`**  
  Jupyter notebooks used to generate the figures and plots presented in the paper.

- **`conf/`**  
  Configuration files defining datasets, models, normalization strategies, and experimental settings.

- **`main_loop.py`**  
  Main script to run experiments and reproduce the results.

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Experiments

To reproduce a single result from the paper, run:

```bash
python eval.py model.normalizer_name=<NORMALIZER_NAME> model.use_asinh=<True/False> dataset.testsets=[<DATASET_NAME1>,<DATASET_NAME2>,...] model.context_length=<CONTEXT_LENGTH>
```

Where: 
- `<NORMALIZER_NAME>`: One of `CausalRevIN`, `RevIN`, or `WURevIN`.  
- `<True/False>`: Whether to use the asinh transformation.  
- `<DATASET_NAME1>,<DATASET_NAME2>,...`: List of dataset names to evaluate on ('gift_eval', 'artificial', 'utsd').  
- `<CONTEXT_LENGTH>`: Context length for the model (Recommend 128, 256, 512, 1024).

To reproduce all results from the paper, run:

```bash
python loop_eval.py
```

⚠️ **Note:** Running all experiments is computationally expensive and may take a long time.  
To run only a subset of experiments, modify the configuration files in the `conf/` folder (e.g., select specific datasets, models, or normalization methods).

---

## Training Strategy

The training strategy implemented in this repository is adapted from the PatchFM framework:

https://github.com/vilhess/PatchFM/

---

## Datasets and Models

Datasets and pretrained models are automatically downloaded at runtime using the Hugging Face `datasets` and `transformers` libraries. No manual download is required.

---

## Reproducibility Notes

- Results in `results/` correspond to the experiments reported in the paper.
- Configuration files in `conf/` fully specify each experimental setup.