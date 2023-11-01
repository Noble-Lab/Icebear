#!/usr/bin/env python

import os, sys, argparse, random;
proj_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'));
sys.path.append(proj_dir)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import tensorflow as tf; #1.13
from tensorflow.python.keras import backend as K
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score
import scipy
from scipy.sparse import csc_matrix, coo_matrix
from scipy.stats import sem
import math
import pandas as pd
import numpy as np;
import anndata as ad
from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import mean_squared_error
#from scipy.stats import ttest_rel, ttest_ind

"""
model.py builds conditional variational autoencoders that integrates scRNA-seq profiles across species that can return several outputs:
1. aligned cell embeddings (with dimension embed_dim) across species
2. cross-species predicted profiles (sequencing-depth normalized gene expression profiles)
"""


def calc_zinb_loss(px_dropout, px_r, px_scale, input_x, reconstr_x):
    softplus_pi = tf.nn.softplus(-px_dropout)  #  uses log(sigmoid(x)) = -softplus(-x)
    log_theta_eps = tf.log(px_r + 1e-8)
    log_theta_mu_eps = tf.log(px_r + reconstr_x + 1e-8)
    pi_theta_log = -px_dropout + tf.multiply(px_r, (log_theta_eps - log_theta_mu_eps))

    case_zero = tf.nn.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = tf.multiply(tf.dtypes.cast(input_x < 1e-8, tf.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + tf.multiply(input_x, (tf.log(reconstr_x + 1e-8) - log_theta_mu_eps))
        + tf.lgamma(input_x + px_r)
        - tf.lgamma(px_r)
        - tf.lgamma(input_x + 1)
    )
    mul_case_non_zero = tf.multiply(tf.dtypes.cast(input_x > 1e-8, tf.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero
    translator_loss_x = - tf.reduce_mean(tf.reduce_sum(res, axis=1))
    return(translator_loss_x)


class RNAAE:
    def __init__(self, input_dim_x, batch_dim_x, embed_dim_x, dispersion, nlayer, dropout_rate, output_model, learning_rate_x, nlabel=1, discriminator_weight=1, epsilon=0.000001, hidden_frac=2, kl_weight=1):
        """
        Network architecture and optimization

        Inputs
        ----------
        input: scRNA expression, ncell x input_dim, float
        batch: scRNA batch factor, ncell x batch_dim, int

        Parameters
        ----------
        kl_weight: non-negative value, float
        input_dim_x: #genes, int
        batch_dim_x: dimension of batch matrix int
        embed_dim_x: embedding dimension in s VAE, int
        learning_rate_x: scRNA VAE learning rate, float
        nlayer: number of hidden layers in encoder/decoder, int, >=1
        dropout_rate: dropout rate in VAE, float
        dispersion: estimate dispersion per gene&batch: "genebatch" or per gene&cell: "genecell" or "genebatch"
        hidden_frac: used to divide intermediate layer dimension to shrink the total paramater size to fit into memory, int
        kl_weight: weight of KL divergence loss in VAE, float
        epsilon: epsilon term of scRNA autoencoder adam optimizer, float
        nlabel: number of dimensions appended at the last of batch factor, that represents one-hot-encoding of species, int
        discriminator_weight: discriminator loss weight, only applicable if discriminator is used in training, float

        """
        self.input_dim_x = input_dim_x;
        self.batch_dim_x = batch_dim_x;
        self.embed_dim_x = embed_dim_x;
        self.learning_rate_x = learning_rate_x;
        self.nlayer = nlayer;
        self.dropout_rate = dropout_rate;
        self.input_x = tf.placeholder(tf.float32, shape=[None, self.input_dim_x]);
        self.batch_x = tf.placeholder(tf.float32, shape=[None, self.batch_dim_x]);
        self.batch_x_decoder = tf.placeholder(tf.float32, shape=[None, self.batch_dim_x]);
        self.kl_weight_x = tf.placeholder(tf.float32, None);
        self.dispersion = dispersion
        self.hidden_frac = hidden_frac
        self.kl_weight = kl_weight
        self.epsilon = epsilon; #float, epsilon term of scRNA autoencoder adam optimizer
        self.nlabel = nlabel

        def encoder_rna(input_data, nlayer, hidden_frac, reuse=tf.AUTO_REUSE):
            """
            scRNA encoder
            Parameters
            ----------
            hidden_frac: used to divide intermediate dimension to shrink the total paramater size to fit into memory
            input_data: generated from tf.concat([self.input_x, self.batch_x], 1), ncells x (input_dim_x + batch_dim_x)
            """
            with tf.variable_scope('encoder_x', reuse=tf.AUTO_REUSE):
                self.intermediate_dim = int(math.sqrt((self.input_dim_x + self.batch_dim_x) * self.embed_dim_x)/hidden_frac)
                l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='encoder_x_0')(input_data);
                l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                l1 = tf.nn.leaky_relu(l1)
                l1 = tf.nn.dropout(l1, rate=self.dropout_rate);

                for layer_i in range(1, nlayer):
                    l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='encoder_x_'+str(layer_i))(l1);
                    l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                    l1 = tf.nn.leaky_relu(l1)
                    l1 = tf.nn.dropout(l1, rate=self.dropout_rate);

                encoder_output_mean = tf.layers.Dense(self.embed_dim_x, activation=None, name='encoder_x_mean')(l1)
                encoder_output_var = tf.layers.Dense(self.embed_dim_x, activation=None, name='encoder_x_var')(l1)
                encoder_output_var = tf.clip_by_value(encoder_output_var, clip_value_min = -2000000, clip_value_max=15)
                encoder_output_var = tf.math.exp(encoder_output_var) + 0.0001
                eps = tf.random_normal((tf.shape(input_data)[0], self.embed_dim_x), 0, 1, dtype=tf.float32)
                encoder_output_z = encoder_output_mean + tf.math.sqrt(encoder_output_var) * eps
                return encoder_output_mean, encoder_output_var, encoder_output_z;
            

        def decoder_rna(encoded_data, nlayer, hidden_frac, reuse=tf.AUTO_REUSE):
            """
            scRNA decoder
            Parameters
            ----------
            hidden_frac: intermadiate layer dim, used hidden_frac to shrink the size to fit into memory
            layer_norm_type: how we normalize layer, don't worry about it now
            encoded_data: generated from concatenation of the encoder output self.encoded_x and batch_x: tf.concat([self.encoded_x, self.batch_x], 1), ncells x (embed_dim_x + batch_dim_x)
            """

            self.intermediate_dim = int(math.sqrt((self.input_dim_x + self.batch_dim_x) * self.embed_dim_x)/hidden_frac); 
            with tf.variable_scope('decoder_x', reuse=tf.AUTO_REUSE):
                l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='decoder_x_0')(encoded_data);
                l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                l1 = tf.nn.leaky_relu(l1)
                for layer_i in range(1, nlayer):
                    l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='decoder_x_'+str(layer_i))(l1);
                    l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                    l1 = tf.nn.leaky_relu(l1)
                px = tf.layers.Dense(self.intermediate_dim, activation=tf.nn.relu, name='decoder_x_px')(l1);
                px_scale = tf.layers.Dense(self.input_dim_x, activation=tf.nn.softmax, name='decoder_x_px_scale')(px);
                px_dropout = tf.layers.Dense(self.input_dim_x, activation=None, name='decoder_x_px_dropout')(px) 
                px_r = tf.layers.Dense(self.input_dim_x, activation=None, name='decoder_x_px_r')(px)
                    
                return px_scale, px_dropout, px_r

        self.libsize_x = tf.reduce_sum(self.input_x, 1)
        
        self.px_z_m, self.px_z_v, self.encoded_x = encoder_rna(tf.concat([self.input_x, self.batch_x], 1), self.nlayer, self.hidden_frac);

        ## scRNA reconstruction
        self.px_scale, self.px_dropout, self.px_r = decoder_rna(tf.concat([self.encoded_x, self.batch_x_decoder], 1), self.nlayer, self.hidden_frac);
        if self.dispersion == '_genebatch':
            self.px_r = tf.layers.Dense(self.input_dim_x, activation=None, name='px_r_genebatch_x')(self.batch_x_decoder[:,-self.nlabel:])

        self.px_r = tf.clip_by_value(self.px_r, clip_value_min = -2000000, clip_value_max=15)
        self.px_r = tf.math.exp(self.px_r)
        self.reconstr_x = tf.transpose(tf.transpose(self.px_scale) *self.libsize_x)
        
        ## scRNA reconstruction (from mean)
        self.px_scale_mean, self.px_dropout_mean, self.px_r_mean = decoder_rna(tf.concat([self.px_z_m, self.batch_x_decoder], 1), self.nlayer, self.hidden_frac);
        if self.dispersion == '_genebatch':
            self.px_r_mean = tf.layers.Dense(self.input_dim_x, activation=None, name='px_r_genebatch_x')(self.batch_x_decoder[:,-self.nlabel:])

        self.reconstr_x_mean = tf.transpose(tf.transpose(self.px_scale_mean) *self.libsize_x)
        
        ## scRNA loss
        # reconstr loss
        self.softplus_pi = tf.nn.softplus(-self.px_dropout)
        self.log_theta_eps = tf.log(self.px_r + 1e-8)
        self.log_theta_mu_eps = tf.log(self.px_r + self.reconstr_x + 1e-8)
        self.pi_theta_log = -self.px_dropout + tf.multiply(self.px_r, (self.log_theta_eps - self.log_theta_mu_eps))

        self.case_zero = tf.nn.softplus(self.pi_theta_log) - self.softplus_pi
        self.mul_case_zero = tf.multiply(tf.dtypes.cast(self.input_x < 1e-8, tf.float32), self.case_zero)

        self.case_non_zero = (
            -self.softplus_pi
            + self.pi_theta_log
            + tf.multiply(self.input_x, (tf.log(self.reconstr_x + 1e-8) - self.log_theta_mu_eps))
            + tf.lgamma(self.input_x + self.px_r)
            - tf.lgamma(self.px_r)
            - tf.lgamma(self.input_x + 1)
        )
        self.mul_case_non_zero = tf.multiply(tf.dtypes.cast(self.input_x > 1e-8, tf.float32), self.case_non_zero)

        self.res = self.mul_case_zero + self.mul_case_non_zero
        self.reconstr_loss_x = - tf.reduce_mean(tf.reduce_sum(self.res, axis=1))

        # KL loss
        self.kld_loss_x = tf.reduce_mean(0.5*(tf.reduce_sum(-tf.math.log(self.px_z_v) + self.px_z_v + tf.math.square(self.px_z_m) -1, axis=1)))

        ## optimizers
        self.train_vars_x = [var for var in tf.trainable_variables() if '_x' in var.name];
        self.loss_x = self.reconstr_loss_x + self.kl_weight_x * self.kld_loss_x * self.kl_weight
        self.optimizer_x = tf.train.AdamOptimizer(learning_rate=self.learning_rate_x, epsilon=self.epsilon).minimize(self.loss_x, var_list=self.train_vars_x );

        configuration = tf.compat.v1.ConfigProto()
        configuration.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=configuration)
        #self.sess = tf.Session();
        self.sess.run(tf.global_variables_initializer());

    def train(self, data_x, batch_x, data_x_val, batch_x_val, nepoch_warmup_x, patience, nepoch_klstart_x, output_model, my_epochs,  batch_size, nlayer, kl_weight, dropout_rate=0, save_model=False):
        """
        train to minimize scRNA-seq loss on training set
        iteratively train generator and discriminator
        early stop when loss doesn't improve for 45 epochs on validation set

        """
        val_reconstr_x_loss_list = [];
        val_kl_x_loss_list = [];
        reconstr_x_loss_list = [];
        kl_x_loss_list = [];
        last_improvement=0

        iter_list = []
        loss_val_check_list = []
        sep_train_index = 1
        saver = tf.train.Saver()
        print(output_model)
        if os.path.exists(output_model+'/mymodel.meta'):
            loss_val_check_best = 0
            tf.reset_default_graph()
            saver = tf.train.import_meta_graph(output_model+'/mymodel.meta')
            saver.restore(self.sess, tf.train.latest_checkpoint(output_model+'/'))
        else:
            if data_x.shape[0] % batch_size >0:
                nbatch_train = data_x.shape[0]//batch_size +1
            else:
                nbatch_train = data_x.shape[0]//batch_size
            if data_x_val.shape[0] % batch_size >0:
                nbatch_val = data_x_val.shape[0]//batch_size +1
            else:
                nbatch_val = data_x_val.shape[0]//batch_size
            for iter in range(1, my_epochs):
                print('iter '+str(iter))
                iter_list.append(iter)
                sys.stdout.flush()
                
                if iter < nepoch_klstart_x:
                    kl_weight_x_update = 0
                else:
                    kl_weight_x_update = min(kl_weight, (iter-nepoch_klstart_x)/float(nepoch_warmup_x))
                
                for batch_id in range(0, nbatch_train):
                    data_x_i = data_x[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x.shape[0]),].todense()
                    batch_x_i = batch_x[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x.shape[0]),]
                    self.sess.run(self.optimizer_x, feed_dict={self.input_x: data_x_i, self.batch_x: batch_x_i, self.batch_x_decoder: batch_x_i, self.kl_weight_x: kl_weight_x_update});

                loss_reconstruct_x = []
                loss_kl_x = []
                for batch_id in range(0, nbatch_train):
                    data_x_i = data_x[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x.shape[0]),].todense()
                    batch_x_i = batch_x[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x.shape[0]),]
                    loss_i, loss_reconstruct_x_i, loss_kl_x_i, loss_kl_x_i = self.get_losses_rna(data_x_i, batch_x_i, batch_x_i, kl_weight_x_update);
                    loss_reconstruct_x.append(loss_reconstruct_x_i)
                    loss_kl_x.append(loss_kl_x_i)
                
                reconstr_x_loss_list.append(np.nanmean(np.array(loss_reconstruct_x)))
                kl_x_loss_list.append(np.nanmean(np.array(loss_kl_x)))

                loss_x_val = []
                loss_reconstruct_x_val = []
                loss_kl_x_val = []
                for batch_id in range(0, nbatch_val):
                    data_x_vali = data_x_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_val.shape[0]),].todense()
                    batch_x_vali = batch_x_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_val.shape[0]),]
                    loss_val_i, loss_reconstruct_x_val_i, loss_kl_x_val_i, loss_kl_x_val_i = self.get_losses_rna(data_x_vali, batch_x_vali, batch_x_vali, kl_weight_x_update);
                    loss_x_val.append(loss_val_i)
                    loss_reconstruct_x_val.append(loss_reconstruct_x_val_i)
                    loss_kl_x_val.append(loss_kl_x_val_i)

                loss_val_check = np.nanmean(np.array(loss_x_val))
                val_reconstr_x_loss_list.append(np.nanmean(np.array(loss_reconstruct_x_val)))
                val_kl_x_loss_list.append(np.nanmean(np.array(loss_kl_x_val)))

                if np.isnan(loss_reconstruct_x_val).any():
                    break
                
                if ((iter + 1) % 1 == 0): # check every epoch
                    print('loss_val_check: '+str(loss_val_check))
                    loss_val_check_list.append(loss_val_check)
                    try:
                        loss_val_check_best
                    except NameError:
                        loss_val_check_best = loss_val_check
                    if loss_val_check < loss_val_check_best:
                        #save_sess = self.sess
                        saver.save(self.sess, output_model+'/mymodel')
                        loss_val_check_best = loss_val_check
                        last_improvement = 0
                    else:
                        last_improvement +=1
                    if len(loss_val_check_list) > 1:
                        ## decide on early stopping
                        stop_decision = last_improvement > patience
                        if stop_decision:
                            tf.reset_default_graph()
                            saver = tf.train.import_meta_graph(output_model+'/mymodel.meta')
                            saver.restore(self.sess, tf.train.latest_checkpoint(output_model+'/'))
                            break

        return iter_list, reconstr_x_loss_list, kl_x_loss_list, kl_x_loss_list, val_reconstr_x_loss_list, val_kl_x_loss_list, val_kl_x_loss_list


    def restore(self, restore_folder):
        """
        Restore the tensorflow graph stored in restore_folder.
        """
        saver = tf.train.Saver()
        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(restore_folder+'/mymodel.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint(restore_folder+'/'))

    def predict_embedding(self, data_x, batch_x):
        """
        return scRNA embeddings
        """
        return self.sess.run(self.px_z_m, feed_dict={self.input_x: data_x, self.batch_x: batch_x, self.batch_x_decoder: batch_x});

    def predict_normexp(self, data_x, batch_x, batch_x_decoder):
        """
        return normalized scRNA profile (normalized, use mean of VAE), translated from batch_x to batch_x_decoder
        """
        return self.sess.run(self.px_scale_mean, feed_dict={self.input_x: data_x, self.batch_x: batch_x, self.batch_x_decoder: batch_x_decoder});
    
    def predict_normexp_sample(self, data_x, batch_x, batch_x_decoder):
        """
        return normalized scRNA profile (normalized, use sampled version from VAE so there's some random noise), translated from batch_x to batch_x_decoder
        """
        return self.sess.run(self.px_scale, feed_dict={self.input_x: data_x, self.batch_x: batch_x, self.batch_x_decoder: batch_x_decoder});
        
    def get_losses_rna(self, data_x, batch_x, batch_x_decoder, kl_weight_x):
        """
        return losses
        """
        return self.sess.run([self.loss_x, self.reconstr_loss_x, self.kld_loss_x, self.kld_loss_x], feed_dict={self.input_x: data_x, self.batch_x: batch_x, self.batch_x_decoder: batch_x_decoder, self.kl_weight_x: kl_weight_x});

    def restore(self, restore_folder):
        """
        Restore the tensorflow graph stored in restore_folder.
        """
        saver = tf.train.Saver()
        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(restore_folder+'/mymodel.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint(restore_folder+'/'))


class RNAAEdis:
    def __init__(self, input_dim_x, batch_dim_x, embed_dim_x, dispersion, nlayer, dropout_rate, output_model, learning_rate_x, nlabel=1, discriminator_weight=1, epsilon=0.000001, hidden_frac=2, kl_weight=1):
        """
        Network architecture and optimization

        Inputs
        ----------
        input: scRNA expression, ncell x input_dim, float
        batch: scRNA batch factor, ncell x batch_dim, int

        Parameters
        ----------
        kl_weight: non-negative value, float
        input_dim_x: #genes, int
        batch_dim_x: dimension of batch matrix int
        embed_dim_x: embedding dimension in s VAE, int
        learning_rate_x: scRNA VAE learning rate, float
        nlayer: number of hidden layers in encoder/decoder, int, >=1
        dropout_rate: dropout rate in VAE, float
        dispersion: estimate dispersion per gene&batch: "genebatch" or per gene&cell: "genecell" or "genebatch"
        hidden_frac: used to divide intermediate layer dimension to shrink the total paramater size to fit into memory, int
        kl_weight: weight of KL divergence loss in VAE, float
        epsilon: epsilon term of scRNA autoencoder adam optimizer, float
        nlabel: number of dimensions appended at the last of batch factor, that represents one-hot-encoding of species, int
        discriminator_weight: discriminator loss weight, only applicable if discriminator is used in training, float

        """

        self.input_dim_x = input_dim_x;
        self.batch_dim_x = batch_dim_x;
        self.embed_dim_x = embed_dim_x;
        self.learning_rate_x = learning_rate_x;
        self.nlayer = nlayer;
        self.dropout_rate = dropout_rate;
        self.input_x = tf.placeholder(tf.float32, shape=[None, self.input_dim_x]);
        self.batch_x = tf.placeholder(tf.float32, shape=[None, self.batch_dim_x]);
        self.batch_x_decoder = tf.placeholder(tf.float32, shape=[None, self.batch_dim_x]);
        self.kl_weight_x = tf.placeholder(tf.float32, None);
        self.dispersion = dispersion
        self.hidden_frac = hidden_frac
        self.kl_weight = kl_weight
        self.epsilon = epsilon; #float, epsilon term of scRNA autoencoder adam optimizer
        self.nlabel = nlabel
        self.discriminator_weight = discriminator_weight

        def encoder_rna(input_data, nlayer, hidden_frac, reuse=tf.AUTO_REUSE):
            """
            scRNA encoder
            Parameters
            ----------
            hidden_frac: used to divide intermediate dimension to shrink the total paramater size to fit into memory
            input_data: generated from tf.concat([self.input_x, self.batch_x], 1), ncells x (input_dim_x + batch_dim_x)
            """
            with tf.variable_scope('encoder_x', reuse=tf.AUTO_REUSE):
                self.intermediate_dim = int(math.sqrt((self.input_dim_x + self.batch_dim_x) * self.embed_dim_x)/hidden_frac)
                l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='encoder_x_0')(input_data);
                l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                l1 = tf.nn.leaky_relu(l1)
                l1 = tf.nn.dropout(l1, rate=self.dropout_rate);

                for layer_i in range(1, nlayer):
                    l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='encoder_x_'+str(layer_i))(l1);
                    l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                    l1 = tf.nn.leaky_relu(l1)
                    l1 = tf.nn.dropout(l1, rate=self.dropout_rate);

                encoder_output_mean = tf.layers.Dense(self.embed_dim_x, activation=None, name='encoder_x_mean')(l1)
                encoder_output_var = tf.layers.Dense(self.embed_dim_x, activation=None, name='encoder_x_var')(l1)
                encoder_output_var = tf.clip_by_value(encoder_output_var, clip_value_min = -2000000, clip_value_max=15)
                encoder_output_var = tf.math.exp(encoder_output_var) + 0.0001
                eps = tf.random_normal((tf.shape(input_data)[0], self.embed_dim_x), 0, 1, dtype=tf.float32)
                encoder_output_z = encoder_output_mean + tf.math.sqrt(encoder_output_var) * eps
                return encoder_output_mean, encoder_output_var, encoder_output_z;
            

        def decoder_rna(encoded_data, nlayer, hidden_frac, reuse=tf.AUTO_REUSE):
            """
            scRNA decoder
            Parameters
            ----------
            hidden_frac: intermadiate layer dim, used hidden_frac to shrink the size to fit into memory
            layer_norm_type: how we normalize layer, don't worry about it now
            encoded_data: generated from concatenation of the encoder output self.encoded_x and batch_x: tf.concat([self.encoded_x, self.batch_x], 1), ncells x (embed_dim_x + batch_dim_x)
            """

            self.intermediate_dim = int(math.sqrt((self.input_dim_x + self.batch_dim_x) * self.embed_dim_x)/hidden_frac); 
            with tf.variable_scope('decoder_x', reuse=tf.AUTO_REUSE):
                l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='decoder_x_0')(encoded_data);
                l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                l1 = tf.nn.leaky_relu(l1)
                for layer_i in range(1, nlayer):
                    l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='decoder_x_'+str(layer_i))(l1);
                    l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                    l1 = tf.nn.leaky_relu(l1)
                px = tf.layers.Dense(self.intermediate_dim, activation=tf.nn.relu, name='decoder_x_px')(l1);
                px_scale = tf.layers.Dense(self.input_dim_x, activation=tf.nn.softmax, name='decoder_x_px_scale')(px);
                px_dropout = tf.layers.Dense(self.input_dim_x, activation=None, name='decoder_x_px_dropout')(px) 
                px_r = tf.layers.Dense(self.input_dim_x, activation=None, name='decoder_x_px_r')(px)#, use_bias=False
                    
                return px_scale, px_dropout, px_r


        def discriminator(input_data, nlayer, nlabel, reuse=tf.AUTO_REUSE):
            """
            discriminator
            Parameters
            ----------
            input_data: the VAE embeddings
            output predicted labels of batch and timepoint
            """
            with tf.variable_scope('discriminator_dx', reuse=tf.AUTO_REUSE):
                l1 = tf.layers.Dense(int(math.sqrt(self.embed_dim_x * nlabel)), activation=None, name='discriminator_dx_0')(input_data);
                l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                l1 = tf.nn.leaky_relu(l1)
                l1 = tf.nn.dropout(l1, rate=self.dropout_rate);

                for layer_i in range(1, nlayer):
                    l1 = tf.layers.Dense(int(math.sqrt(self.embed_dim_x * nlabel)), activation=None, name='discriminator_dx_'+str(layer_i))(l1);
                    l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                    l1 = tf.nn.leaky_relu(l1)
                    l1 = tf.nn.dropout(l1, rate=self.dropout_rate);

                output = tf.layers.Dense(nlabel, activation=None, name='discriminator_dx_output')(l1)
                return output;
            

        self.libsize_x = tf.reduce_sum(self.input_x, 1)
        
        self.px_z_m, self.px_z_v, self.encoded_x = encoder_rna(tf.concat([self.input_x, self.batch_x], 1), self.nlayer, self.hidden_frac);

        ## scRNA reconstruction
        self.px_scale, self.px_dropout, self.px_r = decoder_rna(tf.concat([self.encoded_x, self.batch_x_decoder], 1), self.nlayer, self.hidden_frac);
        if self.dispersion == '_genebatch':
            self.px_r = tf.layers.Dense(self.input_dim_x, activation=None, name='px_r_genebatch_x')(self.batch_x_decoder[:,-self.nlabel:])

        self.px_r = tf.clip_by_value(self.px_r, clip_value_min = -2000000, clip_value_max=15)
        self.px_r = tf.math.exp(self.px_r)
        self.reconstr_x = tf.transpose(tf.transpose(self.px_scale) *self.libsize_x)
        
        ## scRNA reconstruction (from mean)
        self.px_scale_mean, self.px_dropout_mean, self.px_r_mean = decoder_rna(tf.concat([self.px_z_m, self.batch_x_decoder], 1), self.nlayer, self.hidden_frac);
        if self.dispersion == '_genebatch':
            self.px_r_mean = tf.layers.Dense(self.input_dim_x, activation=None, name='px_r_genebatch_x')(self.batch_x_decoder[:,-self.nlabel:])

        self.reconstr_x_mean = tf.transpose(tf.transpose(self.px_scale_mean) *self.libsize_x)
        
        ## scRNA loss
        # reconstr loss
        self.softplus_pi = tf.nn.softplus(-self.px_dropout)
        self.log_theta_eps = tf.log(self.px_r + 1e-8)
        self.log_theta_mu_eps = tf.log(self.px_r + self.reconstr_x + 1e-8)
        self.pi_theta_log = -self.px_dropout + tf.multiply(self.px_r, (self.log_theta_eps - self.log_theta_mu_eps))

        self.case_zero = tf.nn.softplus(self.pi_theta_log) - self.softplus_pi
        self.mul_case_zero = tf.multiply(tf.dtypes.cast(self.input_x < 1e-8, tf.float32), self.case_zero)

        self.case_non_zero = (
            -self.softplus_pi
            + self.pi_theta_log
            + tf.multiply(self.input_x, (tf.log(self.reconstr_x + 1e-8) - self.log_theta_mu_eps))
            + tf.lgamma(self.input_x + self.px_r)
            - tf.lgamma(self.px_r)
            - tf.lgamma(self.input_x + 1)
        )
        self.mul_case_non_zero = tf.multiply(tf.dtypes.cast(self.input_x > 1e-8, tf.float32), self.case_non_zero)

        self.res = self.mul_case_zero + self.mul_case_non_zero
        self.reconstr_loss_x = - tf.reduce_mean(tf.reduce_sum(self.res, axis=1))

        # discriminator loss
        self.input_label = self.batch_x_decoder[:,-self.nlabel:]
        self.output_label = discriminator(self.px_z_m, 1, self.nlabel)
        if self.nlabel==2:
            cce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            self.discriminator_loss = cce(self.input_label, self.output_label) * self.discriminator_weight
        else:
            self.discriminator_loss = tf.compat.v1.losses.softmax_cross_entropy(self.input_label, self.output_label) * self.discriminator_weight

        # KL loss
        self.kld_loss_x = tf.reduce_mean(0.5*(tf.reduce_sum(-tf.math.log(self.px_z_v) + self.px_z_v + tf.math.square(self.px_z_m) -1, axis=1)))

        ## optimizers
        self.train_vars_x = [var for var in tf.trainable_variables() if '_x' in var.name];
        self.train_vars_dx = [var for var in tf.trainable_variables() if '_dx' in var.name];
        print(self.train_vars_dx)
        self.loss_x = self.reconstr_loss_x + self.kl_weight_x * self.kld_loss_x * self.kl_weight
        self.loss_x_generator = self.loss_x - self.discriminator_loss
        self.optimizer_x = tf.train.AdamOptimizer(learning_rate=self.learning_rate_x, epsilon=self.epsilon).minimize(self.loss_x, var_list=self.train_vars_x );
        self.optimizer_dx_discriminator = tf.train.AdamOptimizer(learning_rate=0.0001, epsilon=self.epsilon).minimize(self.discriminator_loss, var_list=self.train_vars_dx );
        self.optimizer_dx_generator = tf.train.AdamOptimizer(learning_rate=0.0001, epsilon=self.epsilon).minimize(self.loss_x_generator, var_list=self.train_vars_x );

        configuration = tf.compat.v1.ConfigProto()
        configuration.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=configuration)
        self.sess.run(tf.global_variables_initializer());


    def train(self, data_x, batch_x, data_x_val, batch_x_val, nepoch_warmup_x, patience, nepoch_klstart_x, output_model, my_epochs,  batch_size, nlayer, kl_weight, dropout_rate=0, save_model=False):
        """
        train to minimize scRNA-seq loss on training set
        early stop when loss doesn't improve for 45 epochs on validation set

        """
        val_reconstr_x_loss_list = [];
        val_kl_x_loss_list = [];
        val_discriminator_x_loss_list = [];
        reconstr_x_loss_list = [];
        kl_x_loss_list = [];
        discriminator_x_loss_list = [];
        last_improvement=0

        iter_list = []
        loss_val_check_list = []
        sep_train_index = 1
        saver = tf.train.Saver()
        print(output_model)

        sub_index = random.sample(range(data_x.shape[0]), data_x_val.shape[0])
        data_x_sub = data_x[sub_index,:]
        batch_x_sub = batch_x[sub_index,:]
        if os.path.exists(output_model+'_sep2/mymodel.meta'):
            loss_val_check_best = 0
            tf.reset_default_graph()
            saver = tf.train.import_meta_graph(output_model+'_sep2/mymodel.meta')
            saver.restore(self.sess, tf.train.latest_checkpoint(output_model+'_sep2/'))
        else:
            if data_x.shape[0] % batch_size >0:
                nbatch_train = data_x.shape[0]//batch_size +1
            else:
                nbatch_train = data_x.shape[0]//batch_size
            if data_x_val.shape[0] % batch_size >0:
                nbatch_val = data_x_val.shape[0]//batch_size +1
            else:
                nbatch_val = data_x_val.shape[0]//batch_size
            if os.path.exists(output_model+'_sep1/mymodel.meta'):
                loss_val_check_best = 0
                tf.reset_default_graph()
                saver = tf.train.import_meta_graph(output_model+'_sep1/mymodel.meta')
                saver.restore(self.sess, tf.train.latest_checkpoint(output_model+'_sep1/'))
                sep_train_index=2
            for iter in range(1, my_epochs):
                print('iter '+str(iter))
                iter_list.append(iter)
                sys.stdout.flush()

                if sep_train_index == 1:
                    if iter < nepoch_klstart_x:
                        kl_weight_x_update = 0
                    else:
                        kl_weight_x_update = min(kl_weight, (iter-nepoch_klstart_x)/float(nepoch_warmup_x))
                    
                    for batch_id in range(0, nbatch_train):
                        data_x_i = data_x[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x.shape[0]),].todense()
                        batch_x_i = batch_x[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x.shape[0]),]
                        self.sess.run(self.optimizer_x, feed_dict={self.input_x: data_x_i, self.batch_x: batch_x_i, self.batch_x_decoder: batch_x_i, self.kl_weight_x: kl_weight_x_update});
                        
                    loss_reconstruct_x = []
                    loss_kl_x = []
                    loss_discriminator_x = []
                    for batch_id in range(0, data_x_sub.shape[0]//batch_size +1):
                        data_x_i = data_x_sub[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_sub.shape[0]),].todense()
                        batch_x_i = batch_x_sub[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_sub.shape[0]),]
                        loss_i, loss_reconstruct_x_i, loss_kl_x_i, loss_discriminator_i = self.get_losses_rna(data_x_i, batch_x_i, batch_x_i, kl_weight_x_update);
                        loss_reconstruct_x.append(loss_reconstruct_x_i)
                        loss_kl_x.append(loss_kl_x_i)
                        loss_discriminator_x.append(loss_discriminator_i)
                    
                    reconstr_x_loss_list.append(np.nanmean(np.array(loss_reconstruct_x)))
                    kl_x_loss_list.append(np.nanmean(np.array(loss_kl_x)))
                    discriminator_x_loss_list.append(np.nanmean(np.array(loss_discriminator_x)))
                    if np.isnan(reconstr_x_loss_list).any():
                        break

                    loss_x_val = []
                    loss_reconstruct_x_val = []
                    loss_kl_x_val = []
                    loss_discriminator_x_val = []
                    for batch_id in range(0, nbatch_val):
                        data_x_vali = data_x_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_val.shape[0]),].todense()
                        batch_x_vali = batch_x_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_val.shape[0]),]
                        loss_val_i, loss_reconstruct_x_val_i, loss_kl_x_val_i, loss_discriminator_val_i = self.get_losses_rna(data_x_vali, batch_x_vali, batch_x_vali, kl_weight_x_update);
                        loss_x_val.append(loss_val_i)
                        loss_reconstruct_x_val.append(loss_reconstruct_x_val_i)
                        loss_kl_x_val.append(loss_kl_x_val_i)
                        loss_discriminator_x_val.append(loss_discriminator_val_i)

                    loss_val_check = np.nanmean(np.array(loss_x_val))
                    val_reconstr_x_loss_list.append(np.nanmean(np.array(loss_reconstruct_x_val)))
                    val_kl_x_loss_list.append(np.nanmean(np.array(loss_kl_x_val)))
                    val_discriminator_x_loss_list.append(np.nanmean(np.array(loss_discriminator_x_val)))

                if sep_train_index == 2:
                    sub_index = []
                    for batch_value in list(range(self.nlabel)):
                        sub_index_batch = list(np.where(batch_x[:,-batch_value-1] == 1)[0])
                        sub_index.extend(random.sample(sub_index_batch, min(len(sub_index_batch), int(data_x_val.shape[0]/2))))
                    random.shuffle(sub_index)
                    data_x_dis = data_x[sub_index,:]
                    batch_x_dis = batch_x[sub_index,:]
                    if iter < nepoch_klstart_x:
                        kl_weight_x_update = 0
                    else:
                        kl_weight_x_update = min(kl_weight, (iter-nepoch_klstart_x)/float(nepoch_warmup_x))
                    
                    ## discriminator
                    for iter_discriminator in range(1):
                        for batch_id in range(0, data_x_dis.shape[0]//batch_size +1):
                            data_x_i = data_x_dis[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_dis.shape[0]),].todense()
                            batch_x_i = batch_x_dis[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_dis.shape[0]),]
                            self.sess.run(self.optimizer_dx_discriminator, feed_dict={self.input_x: data_x_i, self.batch_x: batch_x_i, self.batch_x_decoder: batch_x_i, self.kl_weight_x: kl_weight_x_update});
                            
                    loss_reconstruct_x = []
                    loss_kl_x = []
                    loss_discriminator_x = []
                    for batch_id in range(0, data_x_sub.shape[0]//batch_size +1):
                        data_x_i = data_x_sub[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_sub.shape[0]),].todense()
                        batch_x_i = batch_x_sub[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_sub.shape[0]),]
                        loss_i, loss_reconstruct_x_i, loss_kl_x_i, loss_discriminator_i = self.get_losses_rna(data_x_i, batch_x_i, batch_x_i, kl_weight_x_update);
                        loss_reconstruct_x.append(loss_reconstruct_x_i)
                        loss_kl_x.append(loss_kl_x_i)
                        loss_discriminator_x.append(loss_discriminator_i)
                    
                    reconstr_x_loss_list.append(np.nanmean(np.array(loss_reconstruct_x)))
                    kl_x_loss_list.append(np.nanmean(np.array(loss_kl_x)))
                    discriminator_x_loss_list.append(np.nanmean(np.array(loss_discriminator_x)))

                    loss_x_val = []
                    loss_reconstruct_x_val = []
                    loss_kl_x_val = []
                    loss_discriminator_x_val = []
                    for batch_id in range(0, nbatch_val):
                        data_x_vali = data_x_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_val.shape[0]),].todense()
                        batch_x_vali = batch_x_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_val.shape[0]),]
                        loss_val_i, loss_reconstruct_x_val_i, loss_kl_x_val_i, loss_discriminator_val_i = self.get_losses_rna(data_x_vali, batch_x_vali, batch_x_vali, kl_weight_x_update);
                        loss_x_val.append(loss_val_i-loss_discriminator_val_i)
                        loss_reconstruct_x_val.append(loss_reconstruct_x_val_i)
                        loss_kl_x_val.append(loss_kl_x_val_i)
                        loss_discriminator_x_val.append(loss_discriminator_val_i)

                    loss_val_check = np.nanmean(np.array(loss_x_val))
                    val_reconstr_x_loss_list.append(np.nanmean(np.array(loss_reconstruct_x_val)))
                    val_kl_x_loss_list.append(np.nanmean(np.array(loss_kl_x_val)))
                    val_discriminator_x_loss_list.append(np.nanmean(np.array(loss_discriminator_x_val)))
                    
                    ## generator
                    for batch_id in range(0, data_x_dis.shape[0]//batch_size +1):
                        data_x_i = data_x_dis[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_dis.shape[0]),].todense()
                        batch_x_i = batch_x_dis[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_dis.shape[0]),]
                        self.sess.run(self.optimizer_dx_generator, feed_dict={self.input_x: data_x_i, self.batch_x: batch_x_i, self.batch_x_decoder: batch_x_i, self.kl_weight_x: kl_weight_x_update});
                    
                    loss_reconstruct_x = []
                    loss_kl_x = []
                    loss_discriminator_x = []
                    for batch_id in range(0, data_x_sub.shape[0]//batch_size +1):
                        data_x_i = data_x_sub[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_sub.shape[0]),].todense()
                        batch_x_i = batch_x_sub[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_sub.shape[0]),]
                        loss_i, loss_reconstruct_x_i, loss_kl_x_i, loss_discriminator_i = self.get_losses_rna(data_x_i, batch_x_i, batch_x_i, kl_weight_x_update);
                        loss_reconstruct_x.append(loss_reconstruct_x_i)
                        loss_kl_x.append(loss_kl_x_i)
                        loss_discriminator_x.append(loss_discriminator_i)
                    
                    iter_list.append(iter+0.5)
                    reconstr_x_loss_list.append(np.nanmean(np.array(loss_reconstruct_x)))
                    kl_x_loss_list.append(np.nanmean(np.array(loss_kl_x)))
                    discriminator_x_loss_list.append(np.nanmean(np.array(loss_discriminator_x)))

                    loss_x_val = []
                    loss_reconstruct_x_val = []
                    loss_kl_x_val = []
                    loss_discriminator_x_val = []
                    for batch_id in range(0, nbatch_val):
                        data_x_vali = data_x_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_val.shape[0]),].todense()
                        batch_x_vali = batch_x_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_val.shape[0]),]
                        #batch_x_decoder_vali = batch_x_val_decoder[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_x_val.shape[0]),].todense()
                        loss_val_i, loss_reconstruct_x_val_i, loss_kl_x_val_i, loss_discriminator_val_i = self.get_losses_rna(data_x_vali, batch_x_vali, batch_x_vali, kl_weight_x_update);
                        loss_x_val.append(loss_val_i-loss_discriminator_val_i)
                        loss_reconstruct_x_val.append(loss_reconstruct_x_val_i)
                        loss_kl_x_val.append(loss_kl_x_val_i)
                        loss_discriminator_x_val.append(loss_discriminator_val_i)

                    loss_val_check = np.nanmean(np.array(loss_x_val))
                    val_reconstr_x_loss_list.append(np.nanmean(np.array(loss_reconstruct_x_val)))
                    val_kl_x_loss_list.append(np.nanmean(np.array(loss_kl_x_val)))
                    val_discriminator_x_loss_list.append(np.nanmean(np.array(loss_discriminator_x_val)))


                if ((iter + 1) % 1 == 0): # check every epoch
                    print('loss_val_check: '+str(loss_val_check))
                    loss_val_check_list.append(loss_val_check)
                    try:
                        loss_val_check_best
                    except NameError:
                        loss_val_check_best = loss_val_check
                    if loss_val_check < loss_val_check_best:
                        saver.save(self.sess, output_model+'_sep'+str(sep_train_index)+'/mymodel')
                        loss_val_check_best = loss_val_check
                        last_improvement = 0
                    else:
                        if sep_train_index == 1:
                            last_improvement +=1
                        else:
                            last_improvement +=1
                    if sep_train_index == 2:
                        saver.save(self.sess, output_model+'_sep'+str(sep_train_index)+'/mymodel')
                    
                    if len(loss_val_check_list) > 1:
                        ## decide on early stopping 
                        stop_decision = last_improvement > patience
                        if stop_decision:
                            last_improvement = 0
                            tf.reset_default_graph()
                            saver = tf.train.import_meta_graph(output_model+'_sep'+str(sep_train_index)+'/mymodel.meta')
                            print("No improvement found during the ( require_improvement) last iterations, stopping optimization.")
                            saver.restore(self.sess, tf.train.latest_checkpoint(output_model+'_sep'+str(sep_train_index)+'/'))
                            del loss_val_check_best
                            sep_train_index +=1
                            if sep_train_index > 2:
                                break

        return iter_list, reconstr_x_loss_list, kl_x_loss_list, discriminator_x_loss_list, val_reconstr_x_loss_list, val_kl_x_loss_list, val_discriminator_x_loss_list

    def predict_embedding(self, data_x, batch_x):
        """
        return scRNA embeddings
        """
        return self.sess.run(self.px_z_m, feed_dict={self.input_x: data_x, self.batch_x: batch_x, self.batch_x_decoder: batch_x});

    def predict_normexp(self, data_x, batch_x, batch_x_decoder):
        """
        return normalized scRNA profile (normalized, use mean of VAE), translated from batch_x to batch_x_decoder
        """
        return self.sess.run(self.px_scale_mean, feed_dict={self.input_x: data_x, self.batch_x: batch_x, self.batch_x_decoder: batch_x_decoder});
        
    def get_losses_rna(self, data_x, batch_x, batch_x_decoder, kl_weight_x):
        """
        return scRNA losses
        """
        return self.sess.run([self.loss_x, self.reconstr_loss_x, self.kld_loss_x, self.discriminator_loss], feed_dict={self.input_x: data_x, self.batch_x: batch_x, self.batch_x_decoder: batch_x_decoder, self.kl_weight_x: kl_weight_x});

    def restore(self, restore_folder):
        """
        Restore the tensorflow graph stored in restore_folder.
        """
        saver = tf.train.Saver()
        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(restore_folder+'/mymodel.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint(restore_folder+'/'))


class RNAAEcombined:
    def __init__(self, input_dim, batch_dim, embed_dim, dispersion, nlayer, dropout_rate, output_model, learning_rate, nlabel=1, discriminator_weight=1, epsilon=0.000001, hidden_frac=2, kl_weight=1, dis=''):
        """
        Network architecture and optimization

        Inputs
        ----------
        input: scRNA expression, ncell x input_dim, float
        batch: scRNA batch factor, ncell x batch_dim, int

        Parameters
        ----------
        kl_weight: non-negative value, float
        input_dim: #genes, int
        batch_dim: dimension of batch matrix int
        embed_dim: embedding dimension in s VAE, int
        learning_rate: scRNA VAE learning rate, float
        nlayer: number of hidden layers in encoder/decoder, int, >=1
        dropout_rate: dropout rate in VAE, float
        dispersion: estimate dispersion per gene&batch: "genebatch" or per gene&cell: "genecell" or "genebatch"
        hidden_frac: used to divide intermediate layer dimension to shrink the total paramater size to fit into memory, int
        kl_weight: weight of KL divergence loss in VAE, float
        epsilon: epsilon term of scRNA autoencoder adam optimizer, float
        nlabel: number of dimensions appended at the last of batch factor, that represents one-hot-encoding of species, int
        discriminator_weight: discriminator loss weight, only applicable if discriminator is used in training, float
        dis: whether to use discriminative training: '' or '_dis'

        """
        self.input_dim = input_dim;
        self.batch_dim = batch_dim;
        self.embed_dim = embed_dim;
        self.learning_rate = learning_rate;
        self.nlayer = nlayer;
        self.dropout_rate = dropout_rate;
        self.input = tf.placeholder(tf.float32, shape=[None, self.input_dim]);
        self.batch = tf.placeholder(tf.float32, shape=[None, self.batch_dim]);
        self.batch_decoder = tf.placeholder(tf.float32, shape=[None, self.batch_dim]);
        self.kl_weight = tf.placeholder(tf.float32, None);
        self.dispersion = dispersion;
        self.hidden_frac = hidden_frac;
        self.epsilon = epsilon;
        self.nlabel = nlabel;
        self.dis = dis;
        self.discriminator_weight = discriminator_weight;

        def encoder(input_data, nlayer, hidden_frac, reuse=tf.AUTO_REUSE):
            """
            scRNA encoder
            Parameters
            ----------
            input_data: generated from tf.concat([self.input, self.batch], 1), ncells x (input_dim + batch_dim)
            
            Outputs
            ----------
            embedding later of VAE representing mean (encoder_output_mean), variance (encoder_output_var), and random sampled value (encoder_output_z) based on mean and variance
            """
            with tf.variable_scope('encoder_x', reuse=tf.AUTO_REUSE):
                self.intermediate_dim = int(math.sqrt((self.input_dim + self.batch_dim) * self.embed_dim)/hidden_frac)
                l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='encoder_x_0')(input_data);
                l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                l1 = tf.nn.leaky_relu(l1)
                l1 = tf.nn.dropout(l1, rate=self.dropout_rate);

                for layer_i in range(1, nlayer):
                    l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='encoder_x_'+str(layer_i))(l1);
                    l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                    l1 = tf.nn.leaky_relu(l1)
                    l1 = tf.nn.dropout(l1, rate=self.dropout_rate);

                encoder_output_mean = tf.layers.Dense(self.embed_dim, activation=None, name='encoder_x_mean')(l1)
                encoder_output_var = tf.layers.Dense(self.embed_dim, activation=None, name='encoder_x_var')(l1)
                encoder_output_var = tf.clip_by_value(encoder_output_var, clip_value_min = -2000000, clip_value_max=15)
                encoder_output_var = tf.math.exp(encoder_output_var) + 0.0001
                eps = tf.random_normal((tf.shape(input_data)[0], self.embed_dim), 0, 1, dtype=tf.float32)
                encoder_output_z = encoder_output_mean + tf.math.sqrt(encoder_output_var) * eps
                return encoder_output_mean, encoder_output_var, encoder_output_z;
            

        def decoder(encoded_data, nlayer, hidden_frac, reuse=tf.AUTO_REUSE):
            """
            scRNA decoder
            Parameters
            ----------
            encoded_data: generated from concatenation of the encoder output self.encoded and batch: tf.concat([self.encoded, self.batch], 1), ncells x (embed_dim + batch_dim)

            Outputs
            ----------
            Mean (px_scale), dispersion (px_r) and dropout rate (px_dropout) of zero-inflated negative binomial distribution
            """

            self.intermediate_dim = int(math.sqrt((self.input_dim + self.batch_dim) * self.embed_dim)/hidden_frac); 
            with tf.variable_scope('decoder_x', reuse=tf.AUTO_REUSE):
                l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='decoder_x_0')(encoded_data);
                l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                l1 = tf.nn.leaky_relu(l1)
                for layer_i in range(1, nlayer):
                    l1 = tf.layers.Dense(self.intermediate_dim, activation=None, name='decoder_x_'+str(layer_i))(l1);
                    l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                    l1 = tf.nn.leaky_relu(l1)
                px = tf.layers.Dense(self.intermediate_dim, activation=tf.nn.relu, name='decoder_x_px')(l1);
                px_scale = tf.layers.Dense(self.input_dim, activation=tf.nn.softmax, name='decoder_x_px_scale')(px);
                px_dropout = tf.layers.Dense(self.input_dim, activation=None, name='decoder_x_px_dropout')(px) 
                px_r = tf.layers.Dense(self.input_dim, activation=None, name='decoder_x_px_r')(px)#, use_bias=False
                    
                return px_scale, px_dropout, px_r


        def discriminator(input_data, nlayer, nlabel, reuse=tf.AUTO_REUSE):
            """
            discriminator
            Parameters
            ----------
            input_data: the cVAE embeddings

            Outputs
            ----------
            predicted score of each embedding's probability in each dimension in nlabel vector
            """
            with tf.variable_scope('discriminator_dx', reuse=tf.AUTO_REUSE):
                l1 = tf.layers.Dense(int(math.sqrt(self.embed_dim * nlabel)), activation=None, name='discriminator_dx_0')(input_data);
                l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                l1 = tf.nn.leaky_relu(l1)
                l1 = tf.nn.dropout(l1, rate=self.dropout_rate);

                for layer_i in range(1, nlayer):
                    l1 = tf.layers.Dense(int(math.sqrt(self.embed_dim * nlabel)), activation=None, name='discriminator_dx_'+str(layer_i))(l1);
                    l1 = tf.contrib.layers.layer_norm(inputs=l1, center=True, scale=True);
                    l1 = tf.nn.leaky_relu(l1)
                    l1 = tf.nn.dropout(l1, rate=self.dropout_rate);

                output = tf.layers.Dense(nlabel, activation=None, name='discriminator_dx_output')(l1)
                return output;
            

        self.libsize = tf.reduce_sum(self.input, 1)
        
        self.px_z_m, self.px_z_v, self.encoded = encoder(tf.concat([self.input, self.batch], 1), self.nlayer, self.hidden_frac);

        # scRNA reconstruction (reconstructed from randomly sampled embedding)
        self.px_scale, self.px_dropout, self.px_r = decoder(tf.concat([self.encoded, self.batch_decoder], 1), self.nlayer, self.hidden_frac);
        if self.dispersion == '_genebatch':
            self.px_r = tf.layers.Dense(self.input_dim, activation=None, name='px_x_r_genebatch')(self.batch_decoder[:,-self.nlabel:])

        self.px_r = tf.clip_by_value(self.px_r, clip_value_min = -2000000, clip_value_max=15)
        self.px_r = tf.math.exp(self.px_r)
        self.reconstr = tf.transpose(tf.transpose(self.px_scale) *self.libsize)
        
        # scRNA reconstruction (reconstructed from mean embedding)
        self.px_scale_mean, self.px_dropout_mean, self.px_r_mean = decoder(tf.concat([self.px_z_m, self.batch_decoder], 1), self.nlayer, self.hidden_frac);
        if self.dispersion == '_genebatch':
            self.px_r_mean = tf.layers.Dense(self.input_dim, activation=None, name='px_x_r_genebatch')(self.batch_decoder[:,-self.nlabel:])

        self.reconstr_mean = tf.transpose(tf.transpose(self.px_scale_mean) *self.libsize)

        ## scRNA loss
        # reconstr loss
        self.reconstr_loss = calc_zinb_loss(self.px_dropout, self.px_r, self.px_scale, self.input, self.reconstr)

        # KL loss
        self.kld_loss = tf.reduce_mean(0.5*(tf.reduce_sum(-tf.math.log(self.px_z_v) + self.px_z_v + tf.math.square(self.px_z_m) -1, axis=1)))

        ## optimizers
        self.train_vars = [var for var in tf.trainable_variables() if '_x' in var.name];
        self.loss = self.reconstr_loss + self.kl_weight * self.kld_loss * self.kl_weight

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.epsilon).minimize(self.loss, var_list=self.train_vars );

        ## add discriminator to training
        if self.dis == 'dis':
            self.input_label = self.batch[:,-self.nlabel:]
            self.output_label = discriminator(self.px_z_m, 1, self.nlabel)
            if self.nlabel==2:
                cce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
                self.discriminator_loss = cce(self.input_label, self.output_label) * self.discriminator_weight
            else:
                self.discriminator_loss = tf.compat.v1.losses.softmax_cross_entropy(self.input_label, self.output_label) * self.discriminator_weight

            self.train_vars_dx = [var for var in tf.trainable_variables() if '_dx' in var.name];
            self.loss_generator = self.loss - self.discriminator_loss
            self.optimizer_dx_discriminator = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.epsilon).minimize(self.discriminator_loss, var_list=self.train_vars_dx );
            self.optimizer_dx_generator = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.epsilon).minimize(self.loss_generator, var_list=self.train_vars );

        configuration = tf.compat.v1.ConfigProto()
        configuration.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=configuration)
        self.sess.run(tf.global_variables_initializer());


    def train(self, data, batch, data_val, batch_val, nepoch_warmup, patience, nepoch_klstart, output_model, my_epochs,  batch_size, nlayer, kl_weight, dropout_rate=0, save_model=False):
        """
        train to minimize scRNA-seq loss on training set
        early stop when loss doesn't improve for 45 epochs on validation set

        """
        val_reconstr_loss_list = [];
        val_kl_loss_list = [];
        reconstr_loss_list = [];
        kl_loss_list = [];
        last_improvement=0

        iter_list = []
        loss_val_check_list = []
        sep_train_index = 1
        saver = tf.train.Saver()
        if os.path.exists(output_model+'/mymodel.meta'):
            loss_val_check_best = 0
            tf.reset_default_graph()
            saver = tf.train.import_meta_graph(output_model+'/mymodel.meta')
            saver.restore(self.sess, tf.train.latest_checkpoint(output_model+'/'))
        else:
            if data.shape[0] % batch_size >0:
                nbatch_train = data.shape[0]//batch_size +1
            else:
                nbatch_train = data.shape[0]//batch_size
            if data_val.shape[0] % batch_size >0:
                nbatch_val = data_val.shape[0]//batch_size +1
            else:
                nbatch_val = data_val.shape[0]//batch_size
            for iter in range(1, my_epochs):
                #print('iter '+str(iter))
                iter_list.append(iter)
                sys.stdout.flush()
                
                if iter < nepoch_klstart:
                    kl_weight_update = 0
                else:
                    kl_weight_update = min(kl_weight, (iter-nepoch_klstart)/float(nepoch_warmup))
                
                for batch_id in range(0, nbatch_train):
                    data_i = data[(batch_size*batch_id) : min(batch_size*(batch_id+1), data.shape[0]),].todense()
                    batch_i = batch[(batch_size*batch_id) : min(batch_size*(batch_id+1), data.shape[0]),]
                    self.sess.run(self.optimizer, feed_dict={self.input: data_i, self.batch: batch_i, self.batch_decoder: batch_i, self.kl_weight: kl_weight_update});

                loss_reconstruct = []
                loss_kl = []
                for batch_id in range(0, nbatch_train):
                    data_i = data[(batch_size*batch_id) : min(batch_size*(batch_id+1), data.shape[0]),].todense()
                    batch_i = batch[(batch_size*batch_id) : min(batch_size*(batch_id+1), data.shape[0]),]
                    loss_i, loss_reconstruct_i, loss_kl_i, loss_kl_i = self.get_losses(data_i, batch_i, batch_i, kl_weight_update);
                    loss_reconstruct.append(loss_reconstruct_i)
                    loss_kl.append(loss_kl_i)
                
                reconstr_loss_list.append(np.nanmean(np.array(loss_reconstruct)))
                kl_loss_list.append(np.nanmean(np.array(loss_kl)))

                loss_val = []
                loss_reconstruct_val = []
                loss_kl_val = []
                for batch_id in range(0, nbatch_val):
                    data_vali = data_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_val.shape[0]),].todense()
                    batch_vali = batch_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_val.shape[0]),]
                    loss_val_i, loss_reconstruct_val_i, loss_kl_val_i, loss_kl_val_i = self.get_losses(data_vali, batch_vali, batch_vali, kl_weight_update);
                    loss_val.append(loss_val_i)
                    loss_reconstruct_val.append(loss_reconstruct_val_i)
                    loss_kl_val.append(loss_kl_val_i)

                loss_val_check = np.nanmean(np.array(loss_val))
                val_reconstr_loss_list.append(np.nanmean(np.array(loss_reconstruct_val)))
                val_kl_loss_list.append(np.nanmean(np.array(loss_kl_val)))

                if np.isnan(loss_reconstruct_val).any():
                    print('= error: NA exists in prediction, check')
                    break
                
                if ((iter + 1) % 1 == 0): # check every epoch
                    print('loss_val_check: '+str(loss_val_check))
                    loss_val_check_list.append(loss_val_check)
                    try:
                        loss_val_check_best
                    except NameError:
                        loss_val_check_best = loss_val_check
                    if loss_val_check < loss_val_check_best:
                        #save_sess = self.sess
                        saver.save(self.sess, output_model+'/mymodel')
                        loss_val_check_best = loss_val_check
                        last_improvement = 0
                    else:
                        last_improvement +=1
                    if len(loss_val_check_list) > 1:
                        stop_decision = last_improvement > patience
                        if stop_decision:
                            tf.reset_default_graph()
                            saver = tf.train.import_meta_graph(output_model+'/mymodel.meta')
                            saver.restore(self.sess, tf.train.latest_checkpoint(output_model+'/'))
                            break

        return iter_list, reconstr_loss_list, kl_loss_list, kl_loss_list, val_reconstr_loss_list, val_kl_loss_list, val_kl_loss_list


    def train_dis(self, data, batch, data_val, batch_val, nepoch_warmup, patience, nepoch_klstart, output_model, my_epochs,  batch_size, nlayer, kl_weight, dropout_rate=0, save_model=False):
        """
        train to minimize scRNA-seq loss on training set
        early stop when loss doesn't improve for 45 epochs on validation set

        """
        val_reconstr_loss_list = [];
        val_kl_loss_list = [];
        val_discriminator_loss_list = [];
        reconstr_loss_list = [];
        kl_loss_list = [];
        discriminator_loss_list = [];
        last_improvement=0

        iter_list = []
        loss_val_check_list = []
        sep_train_index = 1
        saver = tf.train.Saver()

        # select a subset of cells in the training set to report training loss, for the concern of running speed
        sub_index = random.sample(range(data.shape[0]), data_val.shape[0])
        data_sub = data[sub_index,:]
        batch_sub = batch[sub_index,:]
        if os.path.exists(output_model+'/mymodel.meta'):
            loss_val_check_best = 0
            tf.reset_default_graph()
            saver = tf.train.import_meta_graph(output_model+'/mymodel.meta')
            saver.restore(self.sess, tf.train.latest_checkpoint(output_model+'/'))
        else:
            if data.shape[0] % batch_size >0:
                nbatch_train = data.shape[0]//batch_size +1
            else:
                nbatch_train = data.shape[0]//batch_size
            if data_val.shape[0] % batch_size >0:
                nbatch_val = data_val.shape[0]//batch_size +1
            else:
                nbatch_val = data_val.shape[0]//batch_size
            for iter in range(1, my_epochs):
                #print('iter '+str(iter))
                iter_list.append(iter)
                sys.stdout.flush()

                # select a subset of cells with equal number of batch/species labels to train discriminator and generator
                sub_index = []
                for batch_value in list(range(self.nlabel)):
                    sub_index_batch = list(np.where(batch[:,-batch_value-1] == 1)[0])
                    sub_index.extend(random.sample(sub_index_batch, min(len(sub_index_batch), 5000)))
                random.shuffle(sub_index)
                data_dis = data[sub_index,:]
                batch_dis = batch[sub_index,:]
                if data_dis.shape[0] % batch_size >0:
                    nbatch_dis = data_dis.shape[0]//batch_size +1
                else:
                    nbatch_dis = data_dis.shape[0]//batch_size

                if iter < nepoch_klstart:
                    kl_weight_update = 0
                else:
                    kl_weight_update = min(kl_weight, (iter-nepoch_klstart)/float(nepoch_warmup))
                
                ## discriminator
                for batch_id in range(0, nbatch_dis):
                    data_i = data_dis[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_dis.shape[0]),].todense()
                    batch_i = batch_dis[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_dis.shape[0]),]
                    self.sess.run(self.optimizer_dx_discriminator, feed_dict={self.input: data_i, self.batch: batch_i, self.batch_decoder: batch_i, self.kl_weight: kl_weight_update});
                
                loss_reconstruct = []
                loss_kl = []
                loss_discriminator = []
                for batch_id in range(0, data_sub.shape[0]//batch_size +1):
                    data_i = data_sub[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_sub.shape[0]),].todense()
                    batch_i = batch_sub[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_sub.shape[0]),]
                    loss_i, loss_reconstruct_i, loss_kl_i, loss_discriminator_i = self.get_losses(data_i, batch_i, batch_i, kl_weight_update);
                    loss_reconstruct.append(loss_reconstruct_i)
                    loss_kl.append(loss_kl_i)
                    loss_discriminator.append(loss_discriminator_i)
                
                reconstr_loss_list.append(np.nanmean(np.array(loss_reconstruct)))
                kl_loss_list.append(np.nanmean(np.array(loss_kl)))
                discriminator_loss_list.append(np.nanmean(np.array(loss_discriminator)))

                loss_val = []
                loss_reconstruct_val = []
                loss_kl_val = []
                loss_discriminator_val = []
                for batch_id in range(0, nbatch_val):
                    data_vali = data_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_val.shape[0]),].todense()
                    batch_vali = batch_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_val.shape[0]),]
                    #batch_decoder_vali = batch_val_decoder[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_val.shape[0]),].todense()
                    loss_val_i, loss_reconstruct_val_i, loss_kl_val_i, loss_discriminator_val_i = self.get_losses(data_vali, batch_vali, batch_vali, kl_weight_update);
                    loss_val.append(loss_val_i) #early stopping based on VAE loss on validation set
                    loss_reconstruct_val.append(loss_reconstruct_val_i)
                    loss_kl_val.append(loss_kl_val_i)
                    loss_discriminator_val.append(loss_discriminator_val_i)

                loss_val_check = np.nanmean(np.array(loss_val))
                val_reconstr_loss_list.append(np.nanmean(np.array(loss_reconstruct_val)))
                val_kl_loss_list.append(np.nanmean(np.array(loss_kl_val)))
                val_discriminator_loss_list.append(np.nanmean(np.array(loss_discriminator_val)))

                ## reconstructor
                train_index = random.sample(list(range(data.shape[0])), data.shape[0])
                data = data[train_index,:]
                batch = batch[train_index,:]
                for batch_id in range(0, data.shape[0]//batch_size):
                    data_i = data[(batch_size*batch_id) : min(batch_size*(batch_id+1), data.shape[0]),].todense()
                    batch_i = batch[(batch_size*batch_id) : min(batch_size*(batch_id+1), data.shape[0]),]
                    self.sess.run(self.optimizer, feed_dict={self.input: data_i, self.batch: batch_i, self.batch_decoder: batch_i, self.kl_weight: kl_weight_update});
                
                ## generators
                for batch_id in range(0, data_dis.shape[0]//batch_size +1):
                    data_i = data_dis[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_dis.shape[0]),].todense()
                    batch_i = batch_dis[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_dis.shape[0]),]
                    self.sess.run(self.optimizer_dx_generator, feed_dict={self.input: data_i, self.batch: batch_i, self.batch_decoder: batch_i, self.kl_weight: kl_weight_update});
                
                loss_reconstruct = []
                loss_kl = []
                loss_discriminator = []
                for batch_id in range(0, data_sub.shape[0]//batch_size +1):
                    data_i = data_sub[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_sub.shape[0]),].todense()
                    batch_i = batch_sub[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_sub.shape[0]),]
                    loss_i, loss_reconstruct_i, loss_kl_i, loss_discriminator_i = self.get_losses(data_i, batch_i, batch_i, kl_weight_update);
                    loss_reconstruct.append(loss_reconstruct_i)
                    loss_kl.append(loss_kl_i)
                    loss_discriminator.append(loss_discriminator_i)
                
                iter_list.append(iter+0.5)
                reconstr_loss_list.append(np.nanmean(np.array(loss_reconstruct)))
                kl_loss_list.append(np.nanmean(np.array(loss_kl)))
                discriminator_loss_list.append(np.nanmean(np.array(loss_discriminator)))

                loss_val = []
                loss_reconstruct_val = []
                loss_kl_val = []
                loss_discriminator_val = []
                for batch_id in range(0, nbatch_val):
                    data_vali = data_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_val.shape[0]),].todense()
                    batch_vali = batch_val[(batch_size*batch_id) : min(batch_size*(batch_id+1), data_val.shape[0]),]
                    loss_val_i, loss_reconstruct_val_i, loss_kl_val_i, loss_discriminator_val_i = self.get_losses(data_vali, batch_vali, batch_vali, kl_weight_update);
                    loss_val.append(loss_val_i) #early stopping based on VAE loss on validation set
                    loss_reconstruct_val.append(loss_reconstruct_val_i)
                    loss_kl_val.append(loss_kl_val_i)
                    loss_discriminator_val.append(loss_discriminator_val_i)

                loss_val_check = np.nanmean(np.array(loss_val))
                val_reconstr_loss_list.append(np.nanmean(np.array(loss_reconstruct_val)))
                val_kl_loss_list.append(np.nanmean(np.array(loss_kl_val)))
                val_discriminator_loss_list.append(np.nanmean(np.array(loss_discriminator_val)))

                if ((iter + 1) % 1 == 0): # check every epoch
                    print('loss_val_check: '+str(loss_val_check))
                    loss_val_check_list.append(loss_val_check)
                    try:
                        loss_val_check_best
                    except NameError:
                        loss_val_check_best = loss_val_check
                    if loss_val_check < loss_val_check_best:
                        saver.save(self.sess, output_model+'/mymodel')
                        loss_val_check_best = loss_val_check
                        last_improvement = 0
                    else:
                        last_improvement +=1
                    
                    if len(loss_val_check_list) > 1:
                        ## decide on early stopping
                        stop_decision = last_improvement > patience
                        if stop_decision:
                            last_improvement = 0
                            tf.reset_default_graph()
                            saver = tf.train.import_meta_graph(output_model+'/mymodel.meta')
                            print("No improvement found during the (patience) last iterations, stopping optimization.")
                            saver.restore(self.sess, tf.train.latest_checkpoint(output_model+'/'))
                            break

        return iter_list, reconstr_loss_list, kl_loss_list, discriminator_loss_list, val_reconstr_loss_list, val_kl_loss_list, val_discriminator_loss_list


    def predict_embedding(self, data, batch):
        """
        return scRNA and scATAC projections on VAE embedding layers 
        """
        return self.sess.run(self.px_z_m, feed_dict={self.input: data, self.batch: batch, self.batch_decoder: batch});
    
    def predict_reconstruction(self, data, batch, batch_decoder):
        """
        return reconstructed and translated scRNA and scATAC profiles (incorporating sequencing-depth)
        """
        return self.sess.run(self.reconstr_mean, feed_dict={self.input: data, self.batch: batch, self.batch_decoder: batch_decoder});
    
    def predict_normexp(self, data, batch, batch_decoder):
        """
        return scRNA rescaled profile (normalized) 
        """
        return self.sess.run(self.px_scale_mean, feed_dict={self.input: data, self.batch: batch, self.batch_decoder: batch_decoder});
    
    def get_losses(self, data, batch, batch_decoder, kl_weight):
        """
        return scRNA reconstruction loss
        """
        return self.sess.run([self.loss, self.reconstr_loss, self.kld_loss, self.kld_loss], feed_dict={self.input: data, self.batch: batch, self.batch_decoder: batch_decoder, self.kl_weight: kl_weight});

    def restore(self, restore_folder):
        """
        Restore the tensorflow graph stored in restore_folder.
        """
        saver = tf.train.Saver()
        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(restore_folder+'/mymodel.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint(restore_folder+'/'))


