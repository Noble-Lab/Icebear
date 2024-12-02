#!/usr/bin/env python

import os, sys, argparse, random;
import tensorflow as tf;
from tensorflow.python.keras import backend as K
import numpy as np;
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.neighbors import NearestNeighbors
import scipy
from scipy.sparse import csc_matrix, coo_matrix
import anndata as ad
import matplotlib.pyplot as plt

sys.path.append('bin/')
from model import *

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import umap

"""
generate_output.py builds separate autoencoders for each domain, and then connects two bottle-neck layers with translator layer. 
Loss = reconstr_loss of each domain + prediction_error of co-assay data from one domain to the other.
"""

def pred_normexp(autoencoder, rna_data_query, rna_data, target_species, dis, output_prefix, batch_size=128):
    """
    predict denoised scRNA-seq profiles that is projected to the target species 
    Parameters
    ----------
    rna_data: original scRNA data
    rna_data_query: original scRNA query data used to predict the corresponding group and batch in the target species
    target_species: the target species to translate the query data to
    dis: indicates whether the discriminator is used in the model
    output_prefix: output prefix

    Output
    ----------
    imputed_rna: ncell x ngenes (the file might be big since it denoises the original data)
    """
    imputed_rna = {}
    # swap species factor to the target encoding
    target_species_encoding = rna_data[rna_data.obs.species==target_species,][:rna_data_query.shape[0],].obsm['encoding'].to_numpy()
    target_encoding = rna_data_query.obsm['encoding'].to_numpy()
    target_encoding[:, :len(rna_data.obs.species.unique())] = target_species_encoding[:, :len(rna_data.obs.species.unique())]
    if dis == 'dis':
        target_encoding[:, -len(rna_data.obs.species.unique()): ] = target_species_encoding[:, -len(rna_data.obs.species.unique()): ]
    for batch_id in range(0, rna_data_query.shape[0]//batch_size +1):
        index_range = list(range(batch_size*batch_id, min(batch_size*(batch_id+1), rna_data_query.shape[0])))
        imputed_rna[batch_id] = autoencoder.predict_normexp(rna_data_query[index_range,:].X.todense(), rna_data_query[index_range,:].obsm['encoding'].to_numpy(), target_encoding[index_range,:]);

    imputed_rna = np.concatenate([v for k,v in sorted(imputed_rna.items())], axis=0)
    np.savetxt(output_prefix +'imputation_'+str(target_species)+ '.txt', imputed_rna, delimiter='\t', fmt='%1.10f')
    


def pred_embedding(autoencoder, rna_data, output_prefix, batch_size=128):
    """
    predict embedding of cells across species
    Parameters
    ----------
    rna_data: scRNA expression anndata format
    output_prefix: output prefix

    Output
    ----------
    encoded_rna: embeddings of each cell in rna_data

    """
    encoded_rna = {};
    for batch_id in range(0, rna_data.shape[0]//batch_size +1):
        index_range = list(range(batch_size*batch_id, min(batch_size*(batch_id+1), rna_data.shape[0])))
        if len(index_range)>0:
            encoded_rna[batch_id] = autoencoder.predict_embedding(rna_data[index_range,:].X.todense(), rna_data[index_range,:].obsm['encoding'].to_numpy());

    encoded_rna = np.concatenate([v for k,v in sorted(encoded_rna.items())], axis=0)
    np.savetxt(output_prefix+'embedding.txt', encoded_rna, delimiter='\t', fmt='%1.5f')

    batch_label = list(rna_data.obs.batch)
    group_label = list(rna_data.obs.group)
    species_label = list(rna_data.obs.species)
    
    sc_rna_combined_embedding = StandardScaler().fit_transform(encoded_rna)
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.001)
    embedding = reducer.fit_transform(sc_rna_combined_embedding)
    umap_embedding_mat = np.concatenate((embedding, np.array(batch_label)[:, None], np.array(group_label)[:, None], np.array(species_label)[:, None]), axis=1)
    
    np.savetxt(output_prefix+'umap.txt', umap_embedding_mat, delimiter='\t', fmt='%s')

    ## plot UMAP
	#os.system('Rscript ./plot_umap.R '+output_prefix)



def plot_loss_epoch(sim_url, dis):
    """
    plot loss per epoch
    Parameters
    ----------
    sim_url: loss file directory prefix

    Output
    ----------
    png file with loss per epoch

    """
    if os.path.exists(sim_url+'_loss_per_epoch.txt'):
        loss_mat = pd.read_csv(sim_url+'_loss_per_epoch.txt', delimiter='\t')
        loss_mat.columns = ['iter','train_reconstr','train_kl','train_discriminator','val_reconstr', 'val_kl', 'val_discriminator']
        iter_list = loss_mat['iter'].values.tolist()
        reconstr_loss_list = loss_mat['train_reconstr'].values.tolist()
        kl_loss_list = loss_mat['train_kl'].values.tolist()
        discriminator_loss_list = loss_mat['train_discriminator'].values.tolist()
        val_reconstr_loss_list = loss_mat['val_reconstr'].values.tolist()
        val_kl_loss_list = loss_mat['val_kl'].values.tolist()
        val_discriminator_loss_list = loss_mat['val_discriminator'].values.tolist()

        fig = plt.figure(figsize = (15,5))
        fig.subplots_adjust(hspace=.4, wspace=.4)
        ax = fig.add_subplot(1,3,1)
        ax.plot(iter_list, reconstr_loss_list, color='blue', marker='.', markersize=0.5, alpha=1, label='train');
        ax.plot(iter_list, val_reconstr_loss_list, color='orange', marker='.', markersize=0.5, alpha=1, label='val');
        plt.legend(loc='upper right')
        plt.title('reconstr loss')

        ax = fig.add_subplot(1,3,2)
        ax.plot(iter_list, kl_loss_list, color='blue', marker='.', alpha=1, label='train');
        ax.plot(iter_list, val_kl_loss_list, color='orange', marker='.', alpha=1, label='val');
        plt.legend(loc='upper right')
        plt.title('KL loss')

        if dis=='dis':
            ax = fig.add_subplot(1,3,3)
            ax.plot(iter_list, discriminator_loss_list, color='blue', marker='.', alpha=1, label='train');
            ax.plot(iter_list, val_discriminator_loss_list, color='orange', marker='.', alpha=1, label='val');
            plt.legend(loc='upper right')
            plt.title('Discriminator loss')

        fig.savefig(sim_url+ '_loss.png')



def convert_batch_to_onehot(input_dataset_list, dataset_list, output_name=''):
    """
    predict embedding of cells across species
    Parameters
    ----------
    rna_data: scRNA expression anndata format
    output_prefix: output prefix

    Output
    ----------
    encoded_rna: embeddings of each cell in rna_data

    """
    dic_dataset_index = {}
    for i in range(len(dataset_list)):
        dic_dataset_index[dataset_list[i]] = i

    indices = np.vectorize(dic_dataset_index.get)(np.array(input_dataset_list))
    indptr = range(len(indices)+1)
    data = np.ones(len(indices))
    matrix_dataset = scipy.sparse.csr_matrix((data, indices, indptr), shape=(len(input_dataset_list), len(dataset_list)))
    return(coo_matrix(matrix_dataset))



def compute_lisi(X, label, perplexity = 30):
    """
    adapted from https://github.com/slowkow/harmonypy/blob/master/harmonypy/lisi.py
    Compute the Local Inverse Simpson Index (LISI) for each column in metadata.

    LISI is a statistic computed for each item (row) in the data matrix X.

    The following example may help to interpret the LISI values.

    Suppose one of the columns in metadata is a categorical variable with 3 categories.

        - If LISI is approximately equal to 3 for an item in the data matrix,
          that means that the item is surrounded by neighbors from all 3
          categories.

        - If LISI is approximately equal to 1, then the item is surrounded by
          neighbors from 1 category.
    
    The LISI statistic is useful to evaluate whether multiple datasets are
    well-integrated by algorithms such as Harmony [1].

    [1]: Korsunsky et al. 2019 doi: 10.1038/s41592-019-0619-0
    """
    n_cells = len(label)
    n_labels = len(label)
    # We need at least 3 * n_neigbhors to compute the perplexity
    knn = NearestNeighbors(n_neighbors = perplexity * 3, algorithm = 'kd_tree').fit(X)
    distances, indices = knn.kneighbors(X)
    # Don't count yourself
    indices = indices[:,1:]
    distances = distances[:,1:]
    # Save the result
    labels = pd.Categorical(label)
    n_categories = len(labels.categories)
    simpson = compute_simpson(distances.T, indices.T, labels, n_categories, perplexity)
    lisi_df = np.mean(1 / simpson)
    return lisi_df



def compute_simpson(
    distances: np.ndarray,
    indices: np.ndarray,
    labels: pd.Categorical,
    n_categories: int,
    perplexity: float,
    tol: float=1e-5
):
    """
    adapted from https://github.com/slowkow/harmonypy/blob/master/harmonypy/lisi.py
    """
    n = distances.shape[1]
    P = np.zeros(distances.shape[0])
    simpson = np.zeros(n)
    logU = np.log(perplexity)
    # Loop through each cell.
    for i in range(n):
        beta = 1
        betamin = -np.inf
        betamax = np.inf
        # Compute Hdiff
        P = np.exp(-distances[:,i] * beta)
        P_sum = np.sum(P)
        if P_sum == 0:
            H = 0
            P = np.zeros(distances.shape[0])
        else:
            H = np.log(P_sum) + beta * np.sum(distances[:,i] * P) / P_sum
            P = P / P_sum
        Hdiff = H - logU
        n_tries = 50
        for t in range(n_tries):
            # Stop when we reach the tolerance
            if abs(Hdiff) < tol:
                break
            # Update beta
            if Hdiff > 0:
                betamin = beta
                if not np.isfinite(betamax):
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if not np.isfinite(betamin):
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2
            # Compute Hdiff
            P = np.exp(-distances[:,i] * beta)
            P_sum = np.sum(P)
            if P_sum == 0:
                H = 0
                P = np.zeros(distances.shape[0])
            else:
                H = np.log(P_sum) + beta * np.sum(distances[:,i] * P) / P_sum
                P = P / P_sum
            Hdiff = H - logU
        # distancesefault value
        if H == 0:
            simpson[i] = -1
        # Simpson's index
        for label_category in labels.categories:
            ix = indices[:,i]
            q = labels[ix] == label_category
            if np.any(q):
                P_sum = np.sum(P[q])
                simpson[i] += P_sum * P_sum
    return simpson
    


def pred(outdir, input_h5ad, sim_url, target_species, target_batch, target_group, group, dispersion, embed_dim, nlayer, dropout_rate, learning_rate, hidden_frac, kl_weight, discriminator_weight, epsilon, patience, my_epochs, nepoch_warmup, nepoch_klstart, batch_size, train, predict, dis=''):
    """
    train/load the Icebear model
    Usage: cross-species alignment, and cross-species prediction on missing cell types/tissues

    Parameters
    ----------
    outdir: output directory
    input_h5ad: input data, in h5ad format. obs variable should include species and batch

    Output
    ----------
    For cross-species alignment and denoising, no data needs to be held out as test set. 
    For cross-species imputation, we hold out a pre-specified cell type to evaluate the performance.
    """
    os.system('mkdir -p '+ outdir)

    ## ======================================
    ## load data and convert factors (e.g., batch, species) to condition vectors in obsm['encoding']

    logging.info('Loading data...')
    
    nsubsample = 2000
    rna_data = ad.read_h5ad(input_h5ad)
    
    ## make species and batch encoding using one-hot encoding
    for obs_i in ['species', 'batch']:
        batch_encoding = pd.DataFrame(convert_batch_to_onehot(list(rna_data.obs[obs_i]), dataset_list=list(rna_data.obs[obs_i].unique())).todense())
        batch_encoding.index = rna_data.obs.index
        if obs_i == 'species':
            rna_data.obsm['encoding'] = batch_encoding
        else:
            rna_data.obsm['encoding'] = pd.concat([rna_data.obsm['encoding'], batch_encoding], axis=1)

    if group != '' and group in rna_data.obs.columns:
        rna_data.obs['group'] = rna_data.obs[group]
    else:
        rna_data.obs['group'] = ''

    ## when needed, include organ as another factor similar to the batch factor, though this function can be achieved by assigning the organ/tissue column as batch
    #if tissuefactor in rna_data.obs.columns:
    #    batch_encoding = pd.DataFrame(convert_batch_to_onehot(list(rna_data.obs[tissuefactor]), dataset_list=list(rna_data.obs[tissuefactor].unique())).todense())
    #    batch_encoding.index = rna_data.obs.index
    #    rna_data.obsm['encoding'] = pd.concat([rna_data.obsm['encoding'], batch_encoding], axis=1)
    
    ## append the one-hot-encoded species encoding for subsequent use
    if dis == 'dis':
        ## add one hot encoded species
        append_encoding = pd.DataFrame(convert_batch_to_onehot(list(rna_data.obs.species), dataset_list=list(rna_data.obs.species.unique())).todense())
        append_encoding.index = rna_data.obs.index
        tmp_encoding = rna_data.obsm['encoding']
        rna_data.obsm['encoding'] = pd.concat([tmp_encoding, append_encoding], axis=1)
        nlabel = append_encoding.shape[1]
        del tmp_encoding
        del append_encoding
    else:
        nlabel = 0
    
    ## ======================================
    ## define train, validation and test
    
    rna_data_test = rna_data[(rna_data.obs.species==target_species) & (rna_data.obs.batch==target_batch) & (rna_data.obs.group==target_group),]
    rna_data_train = rna_data[~ ((rna_data.obs.species==target_species) & (rna_data.obs.batch==target_batch) & (rna_data.obs.group==target_group)),]
    rna_data_train.obs['rank'] = range(0, rna_data_train.shape[0])

    random.seed(101)
    
    rna_data_val_index = random.sample(range(rna_data_train.shape[0]), min(int(rna_data_train.shape[0] * 0.1), nsubsample*10))
    
    rna_data_val = rna_data_train[rna_data_val_index,:]

    ## use all cells that are not assigned to validation set as training set
    rna_data_train = rna_data_train[list(set(range(rna_data_train.shape[0]))-set(rna_data_val_index)),:]

    ## input all data and shuffle indices of train and validation
    train_index = list(range(rna_data_train.shape[0]))
    random.seed(101)
    random.shuffle(train_index)
    data_train = rna_data_train[train_index,:].X.tocsr()
    batch_train = rna_data_train[train_index,:].obsm['encoding'].to_numpy()

    val_index = list(range(rna_data_val.shape[0]))
    random.seed(101)
    random.shuffle(val_index)
    data_val = rna_data_val[val_index,:].X.tocsr()
    batch_val = rna_data_val[val_index,:].obsm['encoding'].to_numpy()

    print('== data imported ==')
    sys.stdout.flush()

    sim_metric_all = []
    save_model=True
    
    ## ======================================
    ## train the model
    if train == 'train':
        logging.info('Training model...')
        tf.reset_default_graph()
        if dis=='':
            autoencoder = RNAAE(input_dim_x=data_train.shape[1], batch_dim_x=batch_train.shape[1], embed_dim_x=embed_dim, dispersion=dispersion, nlayer=nlayer, dropout_rate=dropout_rate, output_model=sim_url, learning_rate_x=learning_rate, nlabel=nlabel, discriminator_weight=discriminator_weight, epsilon=epsilon, hidden_frac=hidden_frac, kl_weight=kl_weight);
        else:
            autoencoder = RNAAEdis(input_dim_x=data_train.shape[1], batch_dim_x=batch_train.shape[1], embed_dim_x=embed_dim, dispersion=dispersion, nlayer=nlayer, dropout_rate=dropout_rate, output_model=sim_url, learning_rate_x=learning_rate, nlabel=nlabel, discriminator_weight=discriminator_weight, epsilon=epsilon, hidden_frac=hidden_frac, kl_weight=kl_weight);
        
        iter_list, reconstr_loss_list, kl_loss_list, discriminator_loss_list, val_reconstr_loss_list, val_kl_loss_list, val_discriminator_loss_list = autoencoder.train(data_train, batch_train, data_val, batch_val, nepoch_warmup, patience, nepoch_klstart, output_model=sim_url, my_epochs=my_epochs, batch_size=batch_size, nlayer=nlayer, kl_weight=kl_weight, dropout_rate=dropout_rate, save_model=save_model);

        ## write loss per epoch
        if len(iter_list)>0:
            fout = open(sim_url+'_loss_per_epoch.txt', 'w')
            for i in range(len(iter_list)):
                fout.write(str(iter_list[i])+'\t'+str(reconstr_loss_list[i])+'\t'+str(kl_loss_list[i])+'\t'+str(discriminator_loss_list[i])+'\t'+str(val_reconstr_loss_list[i])+'\t'+str(val_kl_loss_list[i])+'\t'+str(val_discriminator_loss_list[i])+'\n')

            fout.close()

            plot_loss_epoch(sim_url, dis)
    
        ## calculate cross-species alignment performance on the validation set, this is used to select the best model
        logging.info('Evaluating predictions and get the best performing model...')
        output_prefix = sim_url + '_eval_'
        sc_val_embedding = autoencoder.predict_embedding(rna_data_val.X.todense(), rna_data_val.obsm['encoding'].to_numpy())
        # standardize each dimension of embedding
        sc_val_embedding = (sc_val_embedding - sc_val_embedding.mean(axis=0)) / (sc_val_embedding.std(axis=0))

        # calculate lisi score based on species labels
        species = list(rna_data_val.obs.species.unique())
        val_lisi = compute_lisi(sc_val_embedding, rna_data_val.obs.species.tolist(), perplexity = 30)
        with open (output_prefix+'lisi.txt', 'w') as fp:
            fp.write(str(val_lisi))
        del rna_data_val

    ## ======================================
    ## load trained model for evaluation and downstream analysis
    logging.info('Loading model with dropout_rate=0...')
    tf.reset_default_graph()
    if dis=='':
        autoencoder = RNAAE(input_dim_x=data_train.shape[1], batch_dim_x=batch_train.shape[1], embed_dim_x=embed_dim, dispersion=dispersion, nlayer=nlayer, dropout_rate=0, output_model=sim_url, learning_rate_x=learning_rate, nlabel=nlabel, discriminator_weight=discriminator_weight, epsilon=epsilon, hidden_frac=hidden_frac, kl_weight=kl_weight);
    else:
        autoencoder = RNAAEdis(input_dim_x=data_train.shape[1], batch_dim_x=batch_train.shape[1], embed_dim_x=embed_dim, dispersion=dispersion, nlayer=nlayer, dropout_rate=0, output_model=sim_url, learning_rate_x=learning_rate, nlabel=nlabel, discriminator_weight=discriminator_weight, epsilon=epsilon, hidden_frac=hidden_frac, kl_weight=kl_weight);
    
    iter_list, reconstr_loss_list, kl_loss_list, discriminator_loss_list, val_reconstr_loss_list, val_kl_loss_list, val_discriminator_loss_list = autoencoder.train(data_train, batch_train, data_val, batch_val, nepoch_warmup, patience, nepoch_klstart, output_model=sim_url, my_epochs=my_epochs, batch_size=batch_size, nlayer=nlayer, kl_weight=kl_weight, dropout_rate=dropout_rate, save_model=save_model);

    del data_train
    del batch_train
    del data_val
    del batch_val


    if predict == 'embedding':
        logging.info('Calculating cell embeddings...')
        output_prefix = sim_url + '_pred_'

        ## output embeddings for cells in rna_data
        pred_embedding(autoencoder, rna_data, output_prefix)

    elif predict == 'expression':
        logging.info('Making predictions on gene expression in target species and cell group...')
        rna_data_query = rna_data[(rna_data.obs.species!=target_species) & (rna_data.obs.batch==target_batch) & (rna_data.obs.group==target_group),]
        output_prefix = sim_url + '_pred_'

        ## output normalized expression on target species space based on cells measured in other species
        pred_normexp(autoencoder, rna_data_query, rna_data, target_species, dis, output_prefix)


def main(args):
    outdir = args.outdir
    input_h5ad = args.input_h5ad
    target_species = args.target_species
    target_batch = args.target_batch
    target_group = args.target_group
    group = args.group
    learning_rate = args.learning_rate;
    embed_dim = args.embed_dim;
    dropout_rate = args.dropout_rate;
    nlayer = args.nlayer;
    batch_size = args.batch_size
    patience = args.patience
    my_epochs = args.my_epochs
    nepoch_warmup = args.nepoch_warmup
    nepoch_klstart = args.nepoch_klstart
    dispersion = args.dispersion
    hidden_frac = args.hidden_frac
    kl_weight = args.kl_weight
    epsilon = args.epsilon
    discriminator_weight = args.discriminator_weight
    dis = args.dis
    
    sim_url = args.outdir + 'cross_species_'+ str(nlayer)+ '_lr'+ str(learning_rate)+'_ndim'+str(embed_dim) + dis
    print(sim_url)
    pred(outdir, input_h5ad, sim_url, target_species, target_batch, target_group, group, dispersion, embed_dim, nlayer, dropout_rate, learning_rate, hidden_frac, kl_weight, discriminator_weight, epsilon, patience, my_epochs, nepoch_warmup, nepoch_klstart, batch_size, args.train, args.predict, dis=dis)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description');
    parser.add_argument('--train', type=str, help='"train": train the model from the beginning; "predict": load existing model for downstream prediction', default = 'predict');
    parser.add_argument('--predict', type=str, help='"predict": predict translation ("expression") and alignment ("embedding") on all data', default = 'embedding');

    parser.add_argument('--outdir', type=str, help='outdir', default='./');
    parser.add_argument('--input_h5ad', type=str, help='input_h5ad');
    parser.add_argument('--learning_rate', type=float, help='learning_rate', default=0.001);

    parser.add_argument('--nlayer', type=int, help='nlayer', default=3);
    parser.add_argument('--batch_size', type=int, help='batch size', default=128);
    parser.add_argument('--epsilon', type=float, help='epsilon, used in Adam optimizer in learning scRNA autoencoder', default=0.01);
    parser.add_argument('--my_epochs', type=int, help='maximum number of epochs', default=1000);
    parser.add_argument('--dropout_rate', type=float, help='dropout_rate for hidden layers of autoencoders', default=0.1);
    parser.add_argument('--embed_dim', type=int, help='embed_dim', default=25);
    parser.add_argument('--patience', type=int, help='patience', default=45);
    parser.add_argument('--nepoch_warmup', type=int, help='nepoch_warmup', default=400);
    parser.add_argument('--nepoch_klstart', type=int, help='nepoch_klstart, at which epoch kl weight start to warm up', default=0);
    parser.add_argument('--dispersion', type=str, help='_ or _genebatch', default='_');
    parser.add_argument('--hidden_frac', type=int, help='hidden_frac', default=2);
    parser.add_argument('--kl_weight', type=float, help='kl_weight', default=1);
    parser.add_argument('--discriminator_weight', type=float, help='discriminator_weight', default=1);
    parser.add_argument('--dis', type=str, help='whether to use disciminator to further align species ("dis" if yes, "" if no)', default='');
    parser.add_argument('--target_species', type=str, help='target species to project the data to', default='human');
    parser.add_argument('--target_batch', type=str, help='target batch id to project the data to', default='');
    parser.add_argument('--target_group', type=str, help='target tissue/cell type group to make predictions to', default='');
    parser.add_argument('--group', type=str, help='column name of obs that correspond to cell group information (e.g., tissue/ cell type labels)', default='');

    args = parser.parse_args();
    main(args);
