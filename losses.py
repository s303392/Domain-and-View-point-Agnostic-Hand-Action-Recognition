#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 16:50:33 2020

@author: asabater
"""

import tensorflow as tf


# https://github.com/wangz10/contrastive_loss/blob/master/losses.py
def supervised_nt_xent_loss(temperature=0.07, base_temperature=0.07):
    '''
    Supervised normalized temperature-scaled cross entropy loss. 
    A variant of Multi-class N-pair Loss from (Sohn 2016)
    Later used in SimCLR (Chen et al. 2020, Khosla et al. 2020).
    Implementation modified from: 
        - https://github.com/google-research/simclr/blob/master/objective.py
        - https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    Args:
        z: hidden vector of shape [bsz, n_features].
        y: ground truth of shape [bsz].
    '''
    
    def loss(y, z):
        #y = (etichette di classe)
        #z = (embeddings dei campioni)

        y = y[:, 0]
        batch_size = tf.shape(z)[0]
        contrast_count = 1
        anchor_count = contrast_count
        y = tf.expand_dims(y, -1)

        # mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
        #     has the same class as sample i. Can be asymmetric.
        #la maschera viene utilizzata per identificare i campioni positivi (stessa classe)
        # mask[i,j] = 1 se i e j sono della stessa classe altrimenti = 0.
        mask = tf.cast(tf.equal(y, tf.transpose(y)), tf.float32)

        #calcolo il prodotto scalare tra gli embeddings, diviso per la temperatura
        # questo calcolo misura la similarità tra ogni coppia di campioni nel batch
        # (un valore più alto indica una maggiore similarità)
        anchor_dot_contrast = tf.divide(
            tf.matmul(z, tf.transpose(z)),
            temperature
        )
        # for numerical stability
        logits_max = tf.reduce_max(anchor_dot_contrast, axis=1, keepdims=True)
        logits = anchor_dot_contrast - logits_max
        # tile mask
        #non vogliamo considerare la somiglianza di un campione con se stesso
        logits_mask = tf.ones_like(mask) - tf.eye(batch_size)
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = tf.exp(logits) * logits_mask
        log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True))

        # compute mean of log-likelihood over positive
        mask_sum = tf.reduce_sum(mask, axis=1)

        # Aggiungi un controllo per evitare la divisione per zero
        mean_log_prob_pos = tf.where(
            mask_sum > 0,
            tf.reduce_sum(mask * log_prob, axis=1) / mask_sum,
            tf.zeros_like(mask_sum)
        )

        # calcolo la loss -> cerca di minimizzare le distanze tra ancora e positivo 
        # e questo avviene perché la log-likelihood dei campioni positivi viene MASSIMIZZATA,
        # il che implica che gli embeddings di campioni della stessa classe devono essere più simili 
        # (distanza minore, valore di similarità alta)

        #La loss è definita come il valore negativo della log-likelihood media dei campioni positivi, 
        # quindi MASSIMIZZARE la log-likelihood equivale a minimizzare la loss.
        loss = -(temperature / base_temperature) * mean_log_prob_pos
        loss = tf.reduce_mean(loss)

        #tf.print("anchor_dot_contrast:", anchor_dot_contrast)
        #tf.print("logits:", logits)
        #tf.print("exp_logits:", exp_logits)
        #tf.print("log_prob:", log_prob)
        tf.print("mean_log_prob_pos:", mean_log_prob_pos)
        tf.print("loss:", loss)

        return loss

    return loss
