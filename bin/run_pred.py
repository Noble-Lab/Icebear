#!/usr/bin/env python

import os, sys, argparse, random;
import tensorflow as tf;
from tensorflow.python.keras import backend as K
import numpy as np;
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.linear_model import RidgeCV
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

def pred_normexp(autoencoder, rna_data, outdir, target_species, target_batch, batch_size=128):
    """
    predict denoised scRNA-seq profiles that is projected to a reference (target species and target batch) so that they are directly comparable 
    Parameters
    ----------
    rna_data: scRNA expression anndata format
    target_species: the target species to translate all data to
    target_batch: the target batch to translate all data to
    output_prefix: output prefix

    Output
    ----------
    imputed_rna: ncell x ngenes (the file might be big since it denoises the original data)
    """
    imputed_rna = {}
    for batch_id in range(0, rna_data.shape[0]//batch_size +1):
        index_range = list(range(batch_size*batch_id, min(batch_size*(batch_id+1), rna_data.shape[0])))
        target_encoding = np.tile(rna_data[(rna_data.obs.species==target_species) & (rna_data.obs.batch==target_batch),].obsm['encoding'].to_numpy()[1,:], (len(index_range),1))
        imputed_rna[batch_id] = autoencoder.predict_normexp(rna_data[index_range,:].X.todense(), rna_data[index_range,:].obsm['encoding'].to_numpy(), target_encoding);

    imputed_rna = np.concatenate([v for k,v in sorted(imputed_rna.items())], axis=0)
    np.savetxt(output_prefix +'imputation_'+str(target_species)+str(target_batch)+'.txt', imputed_rna, delimiter='\t', fmt='%1.10f')
    


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
    celltype_label = list(rna_data.obs.celltype)
    species_label = list(rna_data.obs.species)
    
    sc_rna_combined_embedding = StandardScaler().fit_transform(encoded_rna)
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.001)
    embedding = reducer.fit_transform(sc_rna_combined_embedding)
    #print(embedding.shape)
    umap_embedding_mat = np.concatenate((embedding, np.array(batch_label)[:, None], np.array(celltype_label)[:, None], np.array(species_label)[:, None]), axis=1)
    
    np.savetxt(output_prefix+'_umap.txt', umap_embedding_mat, delimiter='\t', fmt='%s')
    if True:
        os.system('Rscript ./plot_umap.R '+output_prefix)



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



def calc_cor_mmd(x, y, nsubsample = 0, logscale=True, norm_x = 'norm', norm_y = '', return_mmd = False, gamma=0):
    """
    return correlation and MMD between original (x) and predicted (y) scRNA px_scale
    gamma: 0: 
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if norm_x == 'norm':
        ## compare with normalized true profile
        lib = x.sum(axis=1, keepdims=True)
        x = x / lib
    if norm_y == 'norm':
        ## compare with normalized true profile
        lib = y.sum(axis=1, keepdims=True)
        y = y / lib
    if logscale:
        x = np.log1p(x)
        y = np.log1p(y)

    if return_mmd is False:
        #if nsubsample >0:
        #    mmd = mmd_rbf(x[np.random.choice(x.shape[0], size=nsubsample, replace=False),], y[np.random.choice(y.shape[0], size=nsubsample, replace=False),])
        #else:
        #    mmd = mmd_rbf(x, y)
        #pearson_r_bulk, pearson_p_bulk = scipy.stats.pearsonr(np.mean(x, axis=0), np.mean(y, axis=0))
        pearson_r_bulk_list = []
        mmd_list = []
        for nrand in range(10):
            mmd = mmd_rbf(x[np.random.choice(x.shape[0], size=nsubsample, replace=False),], y[np.random.choice(y.shape[0], size=nsubsample, replace=False),], gamma=gamma)
            pearson_r_bulk, pearson_p_bulk = scipy.stats.pearsonr(np.mean(x[np.random.choice(x.shape[0], size=nsubsample, replace=False),], axis=0), np.mean(y[np.random.choice(y.shape[0], size=nsubsample, replace=False),], axis=0))
            pearson_r_bulk_list.append(pearson_r_bulk)
            mmd_list.append(mmd)

        #return(pearson_r_bulk, mmd)
        return(pearson_r_bulk_list, mmd_list)
    else:
        if nsubsample >0:
            mmd = mmd_rbf(x[np.random.choice(x.shape[0], size=nsubsample, replace=False),], y[np.random.choice(y.shape[0], size=nsubsample, replace=False),], gamma=gamma)
        else:
            mmd = mmd_rbf(x, y, gamma=gamma)
        return(mmd)



def mmd_rbf(X, Y, gamma=0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    ## calculate mean pairwise euclidean distance
    if gamma == 0:
        sigma = np.median(euclidean_distances(np.concatenate((X, Y))))
        gamma = 0.5/sigma**2
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()



def pred(outdir, input_h5ad, sim_url, target_species, target_batch, dispersion, embed_dim, nlayer, dropout_rate, learning_rate, hidden_frac, kl_weight, discriminator_weight, epsilon, patience, my_epochs, nepoch_warmup, nepoch_klstart, batch_size, train, evaluate, predict, test_species='', test_celltype='', dis=''):
    """
    train/load the Polarbear model
    Parameters
    ----------
    outdir: output directory
    input_h5ad: input data, in h5ad format. obs variable should include species, batch
    test_species: held out species for evaluation purpose
    test_celltype: held out cell type for evaluation purpose
    train: "train" or "", train model
    """
    os.system('mkdir -p '+ outdir)

    ## ======================================
    ## load data and convert batch, species to condition vectors obsm['encoding']
    """
    The script serves as two purposes - cross-species alignment/data denoising, or cross-species prediction on missing cell types/tissues
    In cross-species alignment and denoising, no data needs to be held out as test set. 
    In cross-species imputation, we hold out a pre-specified cell type to evaluate the performance.
    """

    logging.info('Loading data...')
    
    nsubsample = 2000
    rna_data = ad.read_h5ad(input_h5ad)
    
    
    for obs_i in ['species', 'batch']:
        batch_encoding = pd.DataFrame(convert_batch_to_onehot(list(rna_data.obs[obs_i]), dataset_list=list(rna_data.obs[obs_i].unique())).todense())
        batch_encoding.index = rna_data.obs.index
        if obs_i == 'species':
            rna_data.obsm['encoding'] = batch_encoding
        else:
            rna_data.obsm['encoding'] = pd.concat([rna_data.obsm['encoding'], batch_encoding], axis=1)
    rna_data.obs["tissue"] = rna_data.obs.celltype
    
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
    
    rna_data_test = rna_data[(rna_data.obs.species==target_species) & (rna_data.obs.batch==target_batch),]
    rna_data_train = rna_data[~ ((rna_data.obs.species==target_species) & (rna_data.obs.batch==target_batch)),]
    rna_data_train.obs['rank'] = range(0, rna_data_train.shape[0])

    random.seed(101)
    
    rna_data_val_index = random.sample(range(rna_data_train.shape[0]), min(int(rna_data_train.shape[0] * 0.1), nsubsample*10))
    ## define validation set with consideration of dataset info, if available
    #rna_data_val_index = []
    #for dataset_i in rna_data_train.obs.dataset.unique():
        #rna_data_val_index.extend(random.sample(list(rna_data_train.obs.loc[rna_data_train.obs['dataset']==dataset_i]['rank']), min(int(sum(rna_data_train.obs.dataset==dataset_i) * 0.1), nsubsample)))
    
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

    data_test = rna_data_test.X.tocsr()
    batch_test = rna_data_test.obsm['encoding'].to_numpy()

    print('== data imported ==')
    sys.stdout.flush()

    sim_metric_all = []
    save_model=True
    
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
    
    if evaluate=='evaluate' or predict=='predict':
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

    if evaluate == 'evaluate':
        logging.info('Evaluating predictions...')
        ## get MMD based on validation set, this is used to select best model
        output_prefix = sim_url + '_eval_'
        val_mmd_embedding = []
        sc_val_embedding = autoencoder.predict_embedding(rna_data_val.X.todense(), rna_data_val.obsm['encoding'].to_numpy())
        # standardize each dimension of embedding
        sc_val_embedding = (sc_val_embedding - sc_val_embedding.mean(axis=0)) / (sc_val_embedding.std(axis=0))

        # calculate pairwise MMD and get the mean
        species = list(rna_data_val.obs.species.unique())
        for i in range(len(species)-1):
            for j in range(i, len(species)):
                if i<j:
                    sc_mouse_embedding = sc_val_embedding[rna_data_val.obs.species==species[i],:]
                    sc_human_embedding = sc_val_embedding[rna_data_val.obs.species==species[j],:]
                    mmd_embedding = calc_cor_mmd(sc_mouse_embedding, sc_human_embedding, logscale=False, norm_x = '', norm_y = '', return_mmd = True)
                    val_mmd_embedding.append(mmd_embedding)

        val_mmd = [sum(val_mmd_embedding)/len(val_mmd_embedding)]
        np.savetxt(output_prefix+'_mmd.txt', val_mmd, delimiter='\t', fmt='%s')
        del rna_data_val
        del rna_data_test

    if predict == 'embedding':
        logging.info('Making predictions on embeddings...')
        ## output normalized scRNA prediction
        output_prefix = sim_url + '_pred_'

        ## output embeddings on all training & test data
        pred_embedding(autoencoder, rna_data, output_prefix)

    elif predict == 'expression':
        logging.info('Making predictions on gene expression in target species and batch...')
        ## output normalized scRNA prediction
        output_prefix = sim_url + '_pred_'

        ## output normalized expression on target species space (with all species translated to target species)
        pred_normexp(autoencoder, rna_data, output_prefix, target_species, target_batch)


def main(args):
    outdir = args.outdir
    input_h5ad = args.input_h5ad
    target_species = args.target_species
    target_batch = args.target_batch
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
    
    sim_url = args.outdir + 'cross_species_'+ str(nlayer)+ '_lr'+ str(learning_rate)+'_ndim'+str(embed_dim) +args.dis
    print(sim_url)
    pred(outdir, input_h5ad, sim_url, target_species, target_batch, dispersion, embed_dim, nlayer, dropout_rate, learning_rate, hidden_frac, kl_weight, discriminator_weight, epsilon, patience, my_epochs, nepoch_warmup, nepoch_klstart, batch_size, args.train, args.evaluate, args.predict, test_species='', test_celltype='', dis=args.dis)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description');
    parser.add_argument('--train', type=str, help='"train": train the model from the beginning; "predict": load existing model for downstream prediction', default = 'predict');
    parser.add_argument('--evaluate', type=str, help='"evaluate": evaluate translation and alignment performance on the validation set', default = '');
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
    parser.add_argument('--target_species', type=str, help='species to project the data to', default='human');
    parser.add_argument('--target_batch', type=str, help='batch id to project the data to', default='0');

    args = parser.parse_args();
    main(args);
