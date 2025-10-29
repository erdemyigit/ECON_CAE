# ECON_CAE

## Overview
This repository provides code for training and evaluating Quantized Conditional Autoencoders (CAE) as part of the Endcap Concentrator Trigger (ECON-T) project for CMS HGCAL. The code is organized into various scripts for data processing, model training and evaluation as well as representing the learned latent space of the encoder.

## Setup
To set up the environment, create and activate the Conda environment using the provided YAML file:

```bash
conda env create -f environment.yml
conda activate myenv
```

## CAE Description
The Conditional Autoencoder (CAE) consists of a quantized encoder and an unquantized decoder, with additional conditioning in the latent space for known wafer information. Specifically, for HGCAL wafer encoding, the following conditional variables are used:
- eta
- waferu
- waferv
- wafertype (one-hot encoded into 3 possible types)
- sumCALQ
- layers

Altogether, these 8 conditional variables are concatenated with a 16D latent code, resulting in a 24D input to the decoder.


## Preparing the Dataset
Use the `process_data.py` script to prepare the dataset. Below is an example command:

```bash
python process_data.py --opath test_data_saving --num_files 2 --model_per_eLink --biased 0.90 --save_every_n_files 1 --alloc_geom old --use_local --seed 12345
```

Arguments:
- `--opath`: Output directory for saved data.
- `--num_files`: Number of ntuples to preprocess.
- `--model_per_eLink`: Trains a unique CAE per possible eLink allocation.
- `--model_per_bit_config`: Trains a unique CAE per possible bit allocation.
- `--biased`: Resamples the dataset so that n% of the data is signal and (1-n)% is background (specify n as a float).
- `--save_every_n_files`: Number of ntuples to combine per preprocessed output file.
- `--alloc_geom`: The allocation geometry (old, new).
- `--use_local`: If passed, read .root files from local directory (for CMU Rogue01 GPU Cluster only). If not passed, it uses XRootD to get the data from Tier 3.
- `--seed`: If provided, enforces a fixed random seed for consistent shuffling and splitting (reproducible train/test splits).

## Hyperparameter Scan / Training

The hyperparameter scan is implemented in `run_hyperband_search.py` and uses Keras‑Tuner’s Hyperband strategy to:

1. **Search** for the best hyperparameters (learning rate, optimizer, weight decay, batch size, LR scheduler, etc.).
2. **Refine** the best configuration by retraining over multiple random seeds.
3. **Finalize** training on a larger dataset.
4. **Export** both standard Keras models and CMSSW‑compatible versions.

---

### Quick Test Run

Use `--test_run` to exercise the full pipeline in ~5 minutes:

```bash
python run_hyperband_search.py \
  --opath test_hyperband_search \
  --mname search \
  --model_per_eLink \
  --specific_m 5 \
  --data_path /path/to/data \
  --test_run
```

This will create:

```
test_hyperband_search/
└── hyperband_search_5/
    ├── cae_hyperband_base_dir/                    # TensorBoard root for all trials
    ├── log/                                       # Per-trial & final logs
    │   ├── trial_0/
    │   ├── trial_1/
    │   └── final/
    │       ├── best_hyperparameters.csv           # HPs + best_val_loss + best_seed
    │       ├── performance_records.csv            # val_loss vs seed
    │       └── final_val_loss.csv                 # seed vs larger‑dataset loss
    ├── best_model_eLink_5_post_seed_variation/    # Best model + weights
    ├── best_model_eLink_5_post_seed_variation_for_CMSSW/
    ├── best_model_eLink_5_post_seed_variation_larger_dataset/
    └── best_model_eLink_5_post_seed_variation_larger_dataset_for_CMSSW/
```

---

### Full Usage

```bash
python run_hyperband_search.py \
  --opath <output_path>            # root dir for all outputs
  --mname <model_name>             # prefix when exporting models
  --model_per_eLink                 # scan mode: one model per eLink index
  --specific_m <eLink_index>       # which eLink to scan (2, 3, 4, or 5)
  --data_path <path/to/data>       # where your preprocessed datasets live
  [--test_run]                     # quick 2‑epoch smoke test
  [--skip_to_final_model]          # skip HP search, go straight to final training
  [--just_write_best_hyperparameters]  # only dump best HPs to CSV
  [--num_trials <int>]             # how many Hyperband trials (default 50)
  [--num_seeds <int>]              # how many seeds in seed‑search (default 20)
  [--orthogonal_regularization_factor <float>]  # orthogonal regularization factor (<0: regularization factor tunable, 0: no orthogonal regularization, >0: fixed regularization factor)
```

---

### What Happens Under the Hood

1. **Hyperband Search**  
   - **Hyperparameters**  
     - `lr` (log‑uniform between 1e‑5 and 1e‑2)  
     - `optimizer` (`adam` or `lion`)  
     - `weight_decay` (1e‑6 to 1e‑2)  
     - `batch_size` (64, 128, 256, 512, 1024)  
     - `lr_scheduler` (cosine, warm restarts, step decay, exponential decay)  
     - *(optional)* `orthogonal_regularization_factor`  
   - Logs each trial to `cae_hyperband_base_dir` inside your `--opath`.

2. **Seed Variation**  
   - Retrains the best HP configuration over `--num_seeds` random seeds to find the single seed that minimizes validation loss.  
   - Saves per‑seed results to `log/final/performance_records.csv`.

3. **Larger‑Dataset Training**  
   - Takes the best HP + seed, rebuilds the model, and trains for up to 100 epochs on a much larger hold‑out dataset (500 K samples; 25 K if `--test_run`).  
   - Records the final validation loss in `final_val_loss.csv`.

4. **Model Export**  
   - **Keras**: Saved under  
     `best_model_eLink_<m>_post_seed_variation[_larger_dataset]`  
   - **CMSSW**: Converted via  
     `utils.fix_preprocess_CMSSW.save_CMSSW_compatible_model(...)`  
     and saved in parallel folders ending in `_for_CMSSW`.

## File Descriptions
- `process_data.py`: Processes raw data and prepares the dataset.
- `run_hyperband_search.py`: Runs the hyperband search to find the optimal hyperparameters and trains the best configuration on a larger dataset.
- `utils/fix_preprocess_CMSSW.py`: Preprocesses CAE models for CMS Software (CMSSW).
- `utils/graph.py`: Utility functions for graph operations.
- `utils/utils.py`: General utility functions.
- `utils/telescope.py`: Telescope loss function.
- `utils/files.py`: File I/O helper functions.
- `LatentSpace_Visualization_Recipe.ipynb`: Latent space visualization

## CMSSW Processing and the plotting script
Please refer to the [ECON_CAE_Pipeline Document](https://docs.google.com/document/d/1rFesqtG2wraVT74RyEUB6Ck0vqjd2Gn-G8LA32mphLg/edit?tab=t.0)
