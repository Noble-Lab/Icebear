#!/bin/bash

input_h5ad=../data/example.h5ad
outdir=./

## train the model
for learning_rate in 0.001 0.0001
do
	python ./run_pred.py --input_h5ad $input_h5ad --outdir $outdir --train train --predict embedding --learning_rate $learning_rate --group celltype
	python ./run_pred.py --input_h5ad $input_h5ad --outdir $outdir --train train --predict embedding --learning_rate $learning_rate --group celltype --dis dis
done

