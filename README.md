# Cross-species prediction and comparison
![model](https://github.com/Noble-Lab/Icebear/assets/20168747/93b24d7a-07d8-44f5-a1f7-1348a0f1a9b0)


The cross-species model can automatically correct batch, species and other effects (e.g. sex), and be applied to 

1. cross-species imputation/projection

2. cross-species alignment


## Installation:
```conda env create -f environment.yml```


## example run:
```cd bin/```
```bash ./run.sh```


## Input data:

The code takes in h5ad format (ref https://anndata.readthedocs.io/en/latest/generated/anndata.AnnData.html).

The h5ad consists of:

gene expression count matrix (rna_adata.X)

gene annotation (rna_adata.var)

cell annotation (rna_adata.obs): to enable cross-species imputation/alignment and batch correction, rna_adata.obs needs to contain "species" column (e.g. '0' 'human' 'mouse') and "batch" columns (e.g. '1' '0').  

Example input data: ../data/example.h5ad


## Basic usage:

### 1. Cross-species alignment:

```python ./run_pred.py --input_h5ad $input_h5ad --train train --predict embedding```
Where input_h5ad is the path of input h5ad file


### 2. Cross-species imputation: 

For cross-species gene expression prediction, the target species and batch needs to be specified, so that the output gene expression profile translated from all current data to the target batch and species:
`python ./run_pred.py --input_h5ad $input_h5ad --train train --predict expression --target_species 1 --target_batch 0`


## Hyperparameter tuning:
The model is fairly robust to hyperparameters.
There are two main hyperparameters to tune: learning rate (default is 0.001) and whether to use discriminator to further align datasets across species (default is none).
To alter hyperparameters, users can replace input_h5ad file in ```./run.sh``` to do grid search on their own data.

The output mmd score (in "_mmd.txt") can be used to select best model, where models with lower mmd score should perform better cross-species alignment.
