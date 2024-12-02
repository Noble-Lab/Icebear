# Cross-species prediction and comparison
![model](https://github.com/Noble-Lab/Icebear/assets/20168747/93b24d7a-07d8-44f5-a1f7-1348a0f1a9b0)


The cross-species model can automatically correct batch, species and other effects (e.g. sex), and be applied to 

1. cross-species imputation/projection

2. cross-species alignment


## Installation:
Install through conda:
```
conda env create -f environment.yml
conda activate icebear
```

Install through docker (recommended):
```
apptainer pull docker://bearfam/bears
apptainer shell --nv bears_latest.sif
```

## example run:
```
cd bin/
bash ./run.sh
```


## Input data:

The code takes in h5ad format (ref https://anndata.readthedocs.io/en/latest/generated/anndata.AnnData.html).

The h5ad consists of:

gene expression count matrix (rna_adata.X)

gene annotation (rna_adata.var)

cell annotation (rna_adata.obs): to enable cross-species imputation/alignment and batch correction, rna_adata.obs needs to contain a "species" column (e.g. '0' 'human' 'mouse'). Optional columns includes: "batch" columns (e.g. '1' '0') that can represent batch, condition, or organs that the cells are collected from. Columns that represents cell type or tissue information, which can be used on the prediction and validation stage and is indicated using the --group argument.

Example input data: ../data/example.h5ad


## Basic usage:

### 1. Cross-species alignment:

```python ./run_pred.py --input_h5ad $input_h5ad --train train --group celltype --predict embedding```
Where input_h5ad is the path of input h5ad file


### 2. Cross-species imputation: 

For cross-species gene expression prediction, the target species and batch need to be specified so that the output gene expression profile is translated from all current data to the target batch and species:
`python ./run_pred.py --input_h5ad $input_h5ad --train train --predict expression --target_species 1 --target_batch 0 --group celltype`


## Hyperparameter tuning:
The model is fairly robust to hyperparameters.
There are two main hyperparameters to tune: learning rate (the default is 0.001) and whether to use a discriminator to further align datasets across species (the default is none).
To alter hyperparameters, users can replace input_h5ad file in ```./run.sh```for grid search on their own data.

The output mmd score (in "_mmd.txt") can be used to select best model, where models with lower mmd score should perform better cross-species alignment.
