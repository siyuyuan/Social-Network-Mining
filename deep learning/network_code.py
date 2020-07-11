# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:52:56 2020

@author: yuansiyu
"""

import pandas as pd
import numpy as np
from collections import Counter
import pickle
import os
import re
import jieba
import tensorflow as tf
from sklearn.model_selection import train_test_split
import datetime
from tensorflow import keras
from tensorflow.python.ops import summary_ops_v2
import time
import random
from math import log

def get_inputs(user_pre_count, movie_categories_count, title_count, movie_country_count):
    uid = tf.keras.layers.Input(shape=(1,), dtype='int32', name='uid')  
    user_pre = tf.keras.layers.Input(shape=(user_pre_count,), dtype='int32', name='user_pre')

    movie_id = tf.keras.layers.Input(shape=(1,), dtype='int32', name='movie_id') 
    movie_categories = tf.keras.layers.Input(shape=(movie_categories_count,),
                                             dtype='int32', name='movie_categories') 
    movie_titles = tf.keras.layers.Input(shape=(title_count,), dtype='int32', name='movie_titles') 
    movie_avg_rate = tf.keras.layers.Input(shape=(1,), dtype='float', name='movie_avg_rate')
    movie_year = tf.keras.layers.Input(shape=(1,), dtype='int32', name='movie_year')
    movie_runtime = tf.keras.layers.Input(shape=(1,), dtype='int32', name='movie_runtime')
    movie_country = tf.keras.layers.Input(shape=(movie_country_count,), dtype='int32', name='movie_country')
    
    return uid, user_pre, movie_id, movie_categories, movie_titles, movie_avg_rate, movie_year, movie_runtime, movie_country

def get_user_embedding(embed_dim, uid, user_pre, uid_max, uid_pre_max, user_pre_count):
    uid_embed_layer = tf.keras.layers.Embedding(uid_max, embed_dim,
                                                input_length=1, name='uid_embed_layer')(uid)
    
    user_pre_embed_layer = tf.keras.layers.Embedding(uid_pre_max, embed_dim, input_length = user_pre_count, 
                                                     name='user_pre_embed_layer')(user_pre)

    return uid_embed_layer, user_pre_embed_layer


def get_user_feature_layer(embed_dim, uid_embed_layer, user_pre_embed_layer):

    uid_fc_layer = tf.keras.layers.Dense(embed_dim, name="uid_fc_layer",
                                         activation='relu')(uid_embed_layer)
    
    
    user_pre_fc_layer = tf.keras.layers.Dense(embed_dim, name="user_pre_fc_layer",
                                              activation='relu')(user_pre_embed_layer)
    

    user_combine_layer = tf.keras.layers.concatenate([uid_fc_layer, user_pre_fc_layer], 2)  #(?, 1, 64)
    user_combine_layer = tf.keras.layers.Dense(200, activation='tanh')(user_combine_layer)  #(?, 1, 200)

    user_combine_layer_flat = tf.keras.layers.Reshape([200], name="user_combine_layer_flat")(user_combine_layer)
    return user_combine_layer, user_combine_layer_flat

def get_movie_id_embed_layer(movie_id, movie_id_max, embed_dim):
    movie_id_embed_layer = tf.keras.layers.Embedding(movie_id_max, embed_dim,
                                                     input_length=1,
                                                     name='movie_id_embed_layer')(movie_id)
    return movie_id_embed_layer

def get_movie_avg_rate_embed_layer(movie_avg_rate, embed_dim):
    movie_avg_rate_embed_layer = tf.keras.layers.Embedding(10, embed_dim,
                                                     input_length=1,
                                                     name='movie_avg_rate_embed_layer')(movie_avg_rate)
    return movie_avg_rate_embed_layer

def get_movie_categories_layers(movie_categories, movie_categories_max, embed_dim, movie_categories_count):
    movie_categories_embed_layer = tf.keras.layers.Embedding(movie_categories_max,
                                                             embed_dim, input_length=movie_categories_count,
                                                             name='movie_categories_embed_layer')(movie_categories)
    movie_categories_embed_layer = tf.keras.layers.Lambda(lambda layer: tf.reduce_sum(layer,
                                                                                      axis=1,
                                                                                      keepdims=True))(movie_categories_embed_layer)


    return movie_categories_embed_layer

def get_movie_year_embed_layer(movie_year, movie_year_max, embed_dim):
    movie_year_embed_layer = tf.keras.layers.Embedding(movie_year_max, embed_dim,
                                                       input_length=1,
                                                       name='movie_year_embed_layer')(movie_year)
    return movie_year_embed_layer

def get_movie_runtime_embed_layer(movie_runtime, movie_runtime_max, embed_dim):
    movie_runtime_embed_layer = tf.keras.layers.Embedding(movie_runtime_max, embed_dim,
                                                          input_length=1,
                                                          name='movie_runtime_embed_layer')(movie_runtime)
    return movie_runtime_embed_layer

def get_movie_country_layers(movie_country, movie_country_max, embed_dim, movie_country_count):
    movie_country_embed_layer = tf.keras.layers.Embedding(movie_country_max,
                                                          embed_dim, input_length=movie_country_count,
                                                          name='movie_country_embed_layer')(movie_country)
    movie_country_embed_layer = tf.keras.layers.Lambda(lambda layer: tf.reduce_sum(layer,
                                                                                   axis=1,
                                                                                   keepdims=True))(movie_country_embed_layer)


    return movie_country_embed_layer


def get_movie_cnn_layer(movie_titles, movie_title_max, sentences_size):

    movie_title_embed_layer = tf.keras.layers.Embedding(movie_title_max,
                                                        embed_dim,
                                                        input_length=sentences_size,
                                                        name='movie_title_embed_layer')(movie_titles)
    sp = movie_title_embed_layer.shape
    movie_title_embed_layer_expand = tf.keras.layers.Reshape([sp[1], sp[2], 1])(movie_title_embed_layer)

    pool_layer_lst = []
    for window_size in window_sizes:
        conv_layer = tf.keras.layers.Conv2D(filter_num,
                                            (window_size, embed_dim),
                                            1, activation='relu')(movie_title_embed_layer_expand)
        maxpool_layer = tf.keras.layers.MaxPooling2D(pool_size=(sentences_size - window_size + 1 ,1), strides=1)(conv_layer)
        pool_layer_lst.append(maxpool_layer)

    pool_layer = tf.keras.layers.concatenate(pool_layer_lst, 3, name ="pool_layer")  
    max_num = len(window_sizes) * filter_num
    pool_layer_flat = tf.keras.layers.Reshape([1, max_num], name = "pool_layer_flat")(pool_layer)

    dropout_layer = tf.keras.layers.Dropout(dropout_keep, name = "dropout_layer")(pool_layer_flat)
    return pool_layer_flat, dropout_layer

def get_movie_feature_layer(embed_dim, 
                            movie_id_embed_layer, 
                            movie_categories_embed_layer, 
                            dropout_layer, 
                            movie_avg_rate_embed_layer,
                            movie_year_embed_layer,
                            movie_runtime_embed_layer,
                            movie_country_embed_layer):

    movie_id_fc_layer = tf.keras.layers.Dense(embed_dim, 
                                              name="movie_id_fc_layer", 
                                              activation='relu')(movie_id_embed_layer)
    movie_categories_fc_layer = tf.keras.layers.Dense(embed_dim, 
                                                      name="movie_categories_fc_layer", 
                                                      activation='relu')(movie_categories_embed_layer)
    
    movie_avg_rate_fc_layer = tf.keras.layers.Dense(embed_dim, 
                                                      name="movie_avg_rate_fc_layer", 
                                                      activation='relu')(movie_avg_rate_embed_layer)
    movie_runtime_fc_layer = tf.keras.layers.Dense(embed_dim,
                                                   name="movie_runtime_fc_layer",
                                                   activation='relu')(movie_runtime_embed_layer)
    
    movie_year_fc_layer = tf.keras.layers.Dense(embed_dim,
                                                name="movie_year_fc_layer",
                                                activation='relu')(movie_year_embed_layer)
    
    movie_country_fc_layer = tf.keras.layers.Dense(embed_dim,
                                                   name="movie_country_fc_layer",
                                                   activation='relu')(movie_country_embed_layer)


    movie_combine_layer = tf.keras.layers.concatenate([movie_id_fc_layer, 
                                                       movie_categories_fc_layer, 
                                                       movie_avg_rate_fc_layer,
                                                       movie_runtime_fc_layer,
                                                       movie_year_fc_layer,
                                                       movie_country_fc_layer,
                                                       dropout_layer], 2)  
    
    movie_combine_layer = tf.keras.layers.Dense(200, activation='tanh')(movie_combine_layer)

    movie_combine_layer_flat = tf.keras.layers.Reshape([200], 
                                                       name="movie_combine_layer_flat")(movie_combine_layer)
    
    return movie_combine_layer, movie_combine_layer_flat

def get_batches(Xs, ys, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]
        

MODEL_DIR = "./models"
class my_network(object):
    def __init__(self, batch_size, 
                 user_pre_count, movie_categories_count, 
                 title_count, movie_categories_max, 
                 movie_title_max, uid_max, uid_pre_max, 
                 movie_id_max, embed_dim, movie_country_count,
                 movie_runtime_max,movie_country_max,movie_year_max):
        
        self.batch_size = batch_size
        self.best_loss = 9999
        self.losses = {'train': [], 'test': []}

        uid, user_pre, movie_id, movie_categories, movie_titles, movie_avg_rate, movie_year, movie_runtime, movie_country=get_inputs(user_pre_count, 
                                                                                                                                     movie_categories_count, 
                                                                                                                                     title_count, 
                                                                                                                                     movie_country_count)

        uid_embed_layer, user_pre_embed_layer = get_user_embedding(embed_dim, 
                                                                   uid, 
                                                                   user_pre, 
                                                                   uid_max, 
                                                                   uid_pre_max,
                                                                   user_pre_count)

        user_combine_layer, user_combine_layer_flat = get_user_feature_layer(embed_dim, 
                                                                             uid_embed_layer, 
                                                                             user_pre_embed_layer)

        movie_id_embed_layer = get_movie_id_embed_layer(movie_id, movie_id_max, embed_dim)
        
        movie_categories_embed_layer = get_movie_categories_layers(movie_categories, 
                                                                   movie_categories_max, 
                                                                   embed_dim,
                                                                   movie_categories_count)
        
        movie_avg_rate_embed_layer = get_movie_avg_rate_embed_layer(movie_avg_rate, embed_dim)
        
        pool_layer_flat, dropout_layer = get_movie_cnn_layer(movie_titles, 
                                                             movie_title_max, 
                                                             sentences_size)
        
        movie_year_embed_layer = get_movie_year_embed_layer(movie_year, movie_year_max, embed_dim)
        
        movie_runtime_embed_layer = get_movie_runtime_embed_layer(movie_runtime, movie_runtime_max, embed_dim)
        
        movie_country_embed_layer = get_movie_country_layers(movie_country, movie_country_max, embed_dim, movie_country_count)

        movie_combine_layer, movie_combine_layer_flat = get_movie_feature_layer(embed_dim,
                                                                                movie_id_embed_layer,
                                                                                movie_categories_embed_layer,
                                                                                dropout_layer,
                                                                                movie_avg_rate_embed_layer,
                                                                                movie_year_embed_layer,
                                                                                movie_runtime_embed_layer,
                                                                                movie_country_embed_layer)
        inference = tf.keras.layers.Lambda(lambda layer: 
            tf.reduce_sum(layer[0] * layer[1], axis=1), name="inference")((user_combine_layer_flat, 
                                                                           movie_combine_layer_flat))
        
        inference = tf.keras.layers.Lambda(lambda layer: tf.expand_dims(layer, axis=1))(inference)
        
        self.model = tf.keras.Model(
            inputs=[uid, user_pre, movie_id,movie_categories, 
                    movie_titles,movie_avg_rate, movie_year,
                    movie_runtime, movie_country],
            outputs=[inference])

        self.model.summary()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.ComputeLoss = tf.keras.losses.MeanSquaredError()
        self.ComputeMetrics = tf.keras.metrics.MeanAbsoluteError()

        if tf.io.gfile.exists(MODEL_DIR):
            pass
        else:
            tf.io.gfile.makedirs(MODEL_DIR)

        train_dir = os.path.join(MODEL_DIR, 'summaries', 'train')
        test_dir = os.path.join(MODEL_DIR, 'summaries', 'eval')
        
        #self.train_summary_writer = summary_ops_v2.create_file_writer(train_dir, flush_millis=10000)
        #self.test_summary_writer = summary_ops_v2.create_file_writer(test_dir, flush_millis=10000, name='test')

        checkpoint_dir = os.path.join(MODEL_DIR, 'checkpoints')
        self.checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    def compute_loss(self, labels, logits):
        return tf.reduce_mean(tf.keras.losses.mse(labels, logits))

    def compute_metrics(self, labels, logits):
        return tf.keras.metrics.mae(labels, logits)  #


    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model([x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8]], training=True)
            loss = self.ComputeLoss(y, logits)
            self.ComputeMetrics(y, logits)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, logits

    def training(self, features, targets_values, 
                 movie_categories_count,sentences_size, 
                 movie_country_count, epochs=5, log_freq=2000):

        for epoch_i in range(epochs):
            flag = 0
            train_X, test_X, train_y, test_y = train_test_split(features,
                                                                targets_values,
                                                                test_size=0.2,
                                                                random_state=0)

            train_batches = get_batches(train_X, train_y, self.batch_size)
            batch_num = (len(train_X) // self.batch_size)

            train_start = time.time()
            if True:
                avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
                for batch_i in range(batch_num):
                    flag = flag + 1
                    x, y = next(train_batches)
                    categories = np.zeros([self.batch_size, movie_categories_count])
                    for i in range(self.batch_size):
                        categories[i] = x.take(3, 1)[i]

                    titles = np.zeros([self.batch_size, sentences_size])
                    for i in range(self.batch_size):
                        titles[i] = x.take(2, 1)[i]
                        
                        
                    countrys = np.zeros([self.batch_size, movie_country_count])
                    for i in range(self.batch_size):
                        countrys[i] = x.take(7, 1)[i]
                    
                    #uid, user_pre, movie_id,movie_categories, movie_titles,movie_avg_rate, movie_year, movie_runtime, movie_country

                    loss, logits = self.train_step([np.reshape(x.take(0, 1), 
                                                               [self.batch_size, 1]).astype(np.float32),
                                                    np.reshape(x.take(8, 1), 
                                                               [self.batch_size, 1]).astype(np.float32),
                                                    np.reshape(x.take(1, 1), 
                                                               [self.batch_size, 1]).astype(np.float32),
                                                    categories.astype(np.float32),
                                                    titles.astype(np.float32),
                                                    np.reshape(x.take(4, 1), 
                                                               [self.batch_size, 1]).astype(np.float32),
                                                    np.reshape(x.take(5, 1), 
                                                               [self.batch_size, 1]).astype(np.float32),
                                                    np.reshape(x.take(6, 1), 
                                                               [self.batch_size, 1]).astype(np.float32),
                                                    countrys.astype(np.float32)],
                                                    np.reshape(y, [self.batch_size, 1]).astype(np.float32))
                    avg_loss(loss)
                    self.losses['train'].append(loss)

                    if flag % log_freq == 0:
                        print('Epoch {:>3} Loss: {:0.6f} mae: {:0.6f}'.format(epoch_i, loss, 
                              self.ComputeMetrics.result()))
                        avg_loss.reset_states()
                        self.ComputeMetrics.reset_states()

            train_end = time.time()
            print('\nTrain time for epoch {}: {}'.format(epoch_i + 1, train_end - train_start))
            self.testing((test_X, test_y), self.optimizer.iterations, movie_categories_count, 
                         sentences_size, movie_country_count)
            
        self.export_path = os.path.join(MODEL_DIR, 'export')
        tf.saved_model.save(self.model, self.export_path)

    def testing(self, test_dataset, step_num, movie_categories_count, sentences_size, movie_country_count):
        test_X, test_y = test_dataset
        test_batches = get_batches(test_X, test_y, self.batch_size)

        """Perform an evaluation of `model` on the examples from `dataset`."""
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        #         avg_mae = tf.keras.metrics.Mean('mae', dtype=tf.float32)

        batch_num = (len(test_X) // self.batch_size)
        for batch_i in range(batch_num):
            x, y = next(test_batches)
            categories = np.zeros([self.batch_size, movie_categories_count])
            for i in range(self.batch_size):
                categories[i] = x.take(3, 1)[i]

            titles = np.zeros([self.batch_size, sentences_size])
            for i in range(self.batch_size):
                titles[i] = x.take(2, 1)[i]


            countrys = np.zeros([self.batch_size, movie_country_count])
            for i in range(self.batch_size):
                countrys[i] = x.take(7, 1)[i]

            logits = self.model([np.reshape(x.take(0, 1),[self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(8, 1),[self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(1, 1),[self.batch_size, 1]).astype(np.float32),
                                 categories.astype(np.float32),
                                 titles.astype(np.float32),
                                 np.reshape(x.take(4, 1),[self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(5, 1),[self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(6, 1),[self.batch_size, 1]).astype(np.float32),
                                 countrys.astype(np.float32)], 
                                 training=False)
            
            test_loss = self.ComputeLoss(np.reshape(y, [self.batch_size, 1]).astype(np.float32), logits)
            avg_loss(test_loss)
            self.losses['test'].append(test_loss)
            self.ComputeMetrics(np.reshape(y, [self.batch_size, 1]).astype(np.float32), logits)

        print('Model test set loss: {:0.6f} mae: {:0.6f}'.format(avg_loss.result(),
                                                                 self.ComputeMetrics.result()))

        if avg_loss.result() < self.best_loss:
            self.best_loss = avg_loss.result()
            print("best loss = {}".format(self.best_loss))
            self.checkpoint.save(self.checkpoint_prefix)

    def forward(self, xs):
        predictions = self.model(xs)
        return predictions


def rating_movie(my_model, user_id_val, movie_id_val, movie_categories_count, sentences_size, movie_country_count):
    categories = np.zeros([1, movie_categories_count])
    categories[0] = embed_movie.values[movieid2idx[movie_id_val]][2]
    
    titles = np.zeros([1, sentences_size])
    titles[0] = embed_movie.values[movieid2idx[movie_id_val]][1]
    
    country = np.zeros([1, movie_country_count])
    country[0] = embed_movie.values[movieid2idx[movie_id_val]][6]
    #uid, user_pre, movie_id,movie_categories, movie_titles,movie_avg_rate, movie_year, movie_runtime, movie_country
    inference_val = my_model.model([np.reshape(user_pd.values[user_id_val-1][0], [1, 1]),
                                    np.reshape(user_pd.values[user_id_val-1][3], [1, 1]),
                                    np.reshape(embed_movie.values[movieid2idx[movie_id_val]][0], [1, 1]),
                                    categories,
                                    titles,
                                    np.reshape(embed_movie.values[movieid2idx[movie_id_val]][3], [1, 1]),
                                    np.reshape(embed_movie.values[movieid2idx[movie_id_val]][4], [1, 1]),
                                    np.reshape(embed_movie.values[movieid2idx[movie_id_val]][5], [1, 1]),
                                    country], 
                                    training=False)

    return (inference_val.numpy())

def recommend_same_type_movie(movie_id_val):
    
    norm_movie_matrics = tf.sqrt(tf.reduce_sum(tf.square(movie_matrics), 1, keepdims=True))
    normalized_movie_matrics = movie_matrics / norm_movie_matrics
   
    probs_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
    probs_similarity = tf.matmul(probs_embeddings, tf.transpose(normalized_movie_matrics))
    sim = probs_similarity.numpy()

    ls = movies.loc[movies['movie_id'].isin([str(movie_id_val)])]['title']
    for ele in ls:
        movie_name = ele

    print("您看的电影是：{}".format(movie_name))
    print("以下是给您的推荐：")

    p = np.squeeze(sim)
    dic = {}
    for i in range(len(p)):
        dic[p[i]] = i

    sort_ls = sorted(dic.items(),key=lambda x:x[0])
    results = sort_ls[:5]

    i = 0
    for val in results:
        i = i+1
        select_movie_id = embed_movie.iloc[val[1]]['movie_id']
        ls = movies.loc[movies['movie_id'].isin([str(select_movie_id)])]['title']
        for ele in ls:
            movie_name = ele
        print('\n'+'第' + str(i) + '个推荐：'+ movie_name + '\n')
        
def recommend_your_favorite_movie(user_id_val):

    probs_embeddings = (users_matrics[user_id_val-1]).reshape([1, 200])
    probs_similarity = tf.matmul(probs_embeddings, tf.transpose(movie_matrics))
    sim = probs_similarity.numpy()

    print("以下是给您的推荐：")
    p = np.squeeze(sim)
    dic = {}
    for i in range(len(p)):
        dic[p[i]] = i

    sort_ls = sorted(dic.items(),key=lambda x:x[0])
    results = sort_ls[:5]

    i = 0
    for val in results:
        i = i+1
        select_movie_id = embed_movie.iloc[val[1]]['movie_id']
        ls = movies.loc[movies['movie_id'].isin([str(select_movie_id)])]['title']
        for ele in ls:
            movie_name = ele
        print('\n'+'第' + str(i) + '个推荐：'+ movie_name + '\n')
        
if __name__=='__main__':
    
    movies_title = ['number', 'movie_id', 'title',
                'rate', 'rating_people', 'year', 'country', 'directors',
                'writers', 'actors',  'runtime', 'type', 'review_count', 'tags']
    movies = pd.read_csv('movie_db_remove.csv', header=None, names=movies_title)
    movies = movies.dropna(axis = 0)
    movies['year'] = movies['year'].astype(int)
    movies['runtime'] = movies['runtime'].astype(int)
    movies['movie_id'] = movies['movie_id'].astype(int)

    
    
    ls = []
    for ele in movies['movie_id']:
        ls.append(ele)
        
    user_title = ['user_id', 'name', 'rates',
                'tags','wishMovies']
    users = pd.read_csv('user_db.csv', header=None, names=user_title)
    users = users.dropna(axis = 0)
    
    user_ls = []
    movie_ls = []
    rate_ls = []
    rates = users['rates']
    i = 0
    for uid in users['user_id']:
        temp = rates[users['user_id'] == int(uid)]
        rate = eval(temp[uid-1])
        i = i+1
        try:
            for key in rate.keys():
                rate_ls.append(int(rate[key]))
                user_ls.append(uid)
                movie_ls.append(key)
        except:
            continue
            
    user_movie=pd.DataFrame({'user_id':user_ls, 'movie_id':movie_ls, 'rate':rate_ls})
    user_movie['movie_id'] = user_movie['movie_id'].astype(int)
    user_movie = user_movie[user_movie.movie_id.isin (ls)]
    
    id_set = []
    for val in movies['movie_id']:
        if val not in id_set:
            id_set.append(val)
    
    id2int = {val:ii for ii, val in enumerate(set(id_set))}
    id2int['<PAD>'] = 0
    id_map = {val: id2int[val] for ii,val in enumerate(set(movies['movie_id']))}
    
    
    movies['movie_id'] = movies['movie_id'].map(id_map)
    
    user_movie['movie_id'] = user_movie['movie_id'].map(id_map)
    
    
    user_ls = []
    num_ls = []
    tags_ls = []
    tags = users['tags']
    i = 0
    for uid in users['user_id']:
        temp = tags[users['user_id'] == int(uid)]
        tag = eval(temp[uid-1])
        i = i+1
        for key in tag.keys():
            tag[key] = int(tag[key])
        m = sorted(tag.items(), key=lambda item:item[1], reverse=True)
        if len(m) == 0:
            continue
        n = m[0]
        user_ls.append(uid)
        tags_ls.append(n[0])
        num_ls.append(n[1])
        
    user_tag=pd.DataFrame({'user_id':user_ls, 'tags':tags_ls, 'count':num_ls})
    
    type_set = []
    for val in user_tag['tags']:
        temp = jieba.lcut(val)
        for ele in temp:
            if ele not in type_set:
                type_set.append(ele)
    
    
    #type_set.insert(0,'<PAD>')
    use_type2int = {val:ii+1 for ii, val in enumerate(set(type_set))}
    use_type2int['<PAD>']= 0
    
    type_map = {val:[use_type2int[row] for row in jieba.lcut(val)] for ii,val in enumerate(set(user_tag['tags']))}
    
    for key in type_map.keys():
        type_map[key] = int(sum(type_map[key])/len(type_map[key]))
        
    user_tag['tags'] = user_tag['tags'].map(type_map)
    
    type_set = []
    for val in movies['type']:
        temp = val.replace(' ','').split('/')
        for ele in temp:
            if ele not in type_set:
                type_set.append(ele)
    
    
    
    type2int = {val:ii+1 for ii, val in enumerate(set(type_set))}
    type2int['<PAD>'] = 0
    
    type_map = {val:[type2int[row] for row in val.replace(' ','').split('/')] for ii,val in enumerate(set(movies['type']))}
    
    max_len = 0
    for key in type_map.keys():
        temp = len(type_map[key])
        if temp > max_len:
            max_len = temp
            
    for key in type_map:
        for cnt in range( max_len - len(type_map[key])):
            type_map[key].insert(len(type_map[key]) + cnt,type2int['<PAD>'])
    map_type = movies['type'].map(type_map)
    
    title_set = []
    for val in movies['title']:
        for ele in val:
            if ele not in title_set:
                title_set.append(ele)
    
    
    
    title2int = {val:ii for ii, val in enumerate(set(title_set))}
    title2int['<PAD>'] = 0
    
    title_map = {val:[title2int[row] for row in val] for ii,val in enumerate(set(movies['title']))}
    
    max_len = 0
    for key in title_map.keys():
        temp = len(title_map[key])
        if temp > max_len:
            max_len = temp
            
    for key in title_map:
        for cnt in range(max_len - len(title_map[key])):
            title_map[key].insert(len(title_map[key]) + cnt,title2int['<PAD>'])
    map_title = movies['title'].map(title_map)
    
    
    
    country_set = []
    for val in movies['country']:
        temp = val.replace(' ','').split('/')
        for ele in temp:
            if ele not in country_set:
                country_set.append(ele)
    
    country2int = {val:ii for ii, val in enumerate(set(country_set))}
    country2int['<PAD>'] = 0
    
    country_map = {val:[country2int[row] for row in val.replace(' ','').split('/')] for ii,
                        val in enumerate(set(movies['country']))}
    
    max_len = 0
    for key in country_map.keys():
        temp = len(country_map[key])
        if temp > max_len:
            max_len = temp
            
            
    for key in country_map:
        for cnt in range(max_len - len(country_map[key])):
            country_map[key].insert(len(country_map[key]) + cnt,country2int['<PAD>'])
    map_country = movies['country'].map(country_map)
    
    
    embed_movie=pd.DataFrame({'movie_id':movies['movie_id'], 'title':map_title,
                              'type':map_type, 'avg_rate':movies['rate'], 
                              'year':movies['year'],'runtime':movies['runtime'],
                              'country':map_country})
        
    embed_movie['movie_id'] = embed_movie['movie_id'].astype(int)
    user_movie['movie_id'] = user_movie['movie_id'].astype(int)
    user_movie['user_id'] = user_movie['user_id'].astype(int)
    user_tag['user_id'] = user_tag['user_id'].astype(int)
    user_pd = pd.merge(user_movie,user_tag)
    data = pd.merge(pd.merge(user_movie, embed_movie),user_tag)
    
    data = data.drop(['count'], axis=1)
    target_fields = ['rate']
    features_pd, targets_pd = data.drop(target_fields, axis=1), data[target_fields]
    features_pd['user_id'] = features_pd['user_id'].astype(int)
    features_pd['movie_id'] = features_pd['movie_id'].astype(int)
    features_pd['avg_rate'] = features_pd['avg_rate'].astype(float)
    
    features = features_pd.values
    targets_values = targets_pd.values
    
    
    num_users = users.shape[0]
    num_movie = movies.shape[0]
    num_movie_type = len(type2int)
    title_count = len(features[0][2])
    user_pre_count = 1
    movie_categories_count = len(features[0][3])
    movie_country_count = len(features[0][7])
    
    embed_dim = 32
    uid_max = num_users
    uid_pre_max = len(use_type2int)
    
    movie_year_max = 2020
    movie_runtime_max = 240
    movie_id_max = num_movie
    movie_categories_max =num_movie_type
    movie_title_max = len(title_set) 
    movie_country_max = len(country2int)
    
    
    combiner = "sum"
    sentences_size = title_count
    
    window_sizes = {2, 3, 4, 5}
    
    filter_num = 8
    
    movieid2idx = {val[0]:i for i, val in enumerate(embed_movie.values)}
    
    # Number of Epochs
    num_epochs = 2
    # Batch Size
    batch_size = 128
    
    dropout_keep = 0.5
    # Learning Rate
    learning_rate = 1e-3
    # Show stats for every n number of batches
    show_every_n_batches = 20
    save_dir = './save'
    
    my_net=my_network(batch_size, 
                 user_pre_count, movie_categories_count, 
                 title_count, movie_categories_max, 
                 movie_title_max, uid_max, uid_pre_max, 
                 movie_id_max, embed_dim, movie_country_count,
                 movie_runtime_max,movie_country_max,movie_year_max)
    
    my_net.training(features, targets_values, movie_categories_count, 
                    sentences_size, movie_country_count, epochs=num_epochs)
    
    print('\n'+'---Training is over.We can use to compute the ndcg---'+'\n')
    
    mid_ls = embed_movie['movie_id'].values
    total_ndcg = []
    for ele in users['user_id']:
        l = len(mid_ls)-1
        user_score_dic = {}
        real_score_dic = {}
        for i in range(5):
            index = random.randint(0,l)
            mid = mid_ls[index]
            user_rate = rating_movie(my_net, ele, mid, movie_categories_count, sentences_size, movie_country_count)
            user_rate = user_rate.tolist()[0][0]
            ls = embed_movie.loc[embed_movie['movie_id'].isin([mid])]['avg_rate']
            for temp in ls:
                real_score = temp
            user_score_dic[user_rate] = mid
            real_score_dic[mid] = real_score
            
        user_score_ls = sorted(user_score_dic.items(),key=lambda x:x[0], reverse=True)
        real_score_ls = sorted(real_score_dic.items(),key=lambda x:x[1], reverse=True)
        
        #compute idcg
        idcg = 0
        for i in range(len(real_score_ls)):
            rel = real_score_ls[i][1]
            idcg = idcg + (2**rel - 1) / log(i+2,2)
        
        dcg = 0
        for i in range(len(user_score_ls)):
            mid = user_score_ls[i][1]
            rel = real_score_dic[mid]
            dcg = dcg + (2**rel - 1) / log(i+2,2)
        
        ndcg = dcg/idcg
        total_ndcg.append(ndcg)
        
    total_ndcg_value = sum(total_ndcg) * 1.0 / len(total_ndcg)
    
    print('ndcg is {:.2f}'.format(total_ndcg_value))
    
    
    #uid, user_pre, movie_id, movie_categories, movie_titles, movie_avg_rate
    movie_layer_model = keras.models.Model(inputs=[my_net.model.input[2], 
                                                   my_net.model.input[3], 
                                                   my_net.model.input[4],
                                                   my_net.model.input[5],
                                                   my_net.model.input[6],
                                                   my_net.model.input[7],
                                                   my_net.model.input[8]], 
                                           outputs=my_net.model.get_layer("movie_combine_layer_flat").output)
    movie_matrics = []
    movie_ls = embed_movie.values
    count = 0
    for item in movie_ls:
        categories = np.zeros([1, movie_categories_count])
        categories[0] = item.take(2)
    
        titles = np.zeros([1, sentences_size])
        titles[0] = item.take(1)
        
        country = np.zeros([1, movie_country_count])
        country[0] = item.take(6)
    
        movie_combine_layer_flat_val = movie_layer_model([np.reshape(item.take(0), [1, 1]), 
                                                          categories, 
                                                          titles, 
                                                          np.reshape(item.take(3), [1, 1]),
                                                          np.reshape(item.take(4), [1, 1]),
                                                          np.reshape(item.take(5), [1, 1]),
                                                          country]) 
        movie_matrics.append(movie_combine_layer_flat_val)
        count = count + 1
        print('\r the movie_matrics process:{:.2f}%'.format(count*100/len(movie_ls)),end='')
    
    pickle.dump((np.array(movie_matrics).reshape(-1, 200)), open('movie_matrics.p', 'wb'))

    print('\n'+'Have written the movie_matrics'+'\n')
    #uid, user_pre, movie_id, movie_categories, movie_titles, movie_avg_rate
    user_layer_model = keras.models.Model(inputs=[my_net.model.input[0], 
                                                  my_net.model.input[1]], 
                                          outputs=my_net.model.get_layer("user_combine_layer_flat").output)
    users_matrics = []
    users_ls = user_pd.values
    count = 0
    for item in users_ls:
    
        user_combine_layer_flat_val = user_layer_model([np.reshape(item.take(0), [1, 1]), 
                                                        np.reshape(item.take(3), [1, 1])])  
        users_matrics.append(user_combine_layer_flat_val)
        count = count + 1
        print('\r the users_matrics process:{:.2f}%'.format(count*100/len(users_ls)),end='')
    
    pickle.dump((np.array(users_matrics).reshape(-1, 200)), open('users_matrics.p', 'wb'))
    
    print('\n'+'Have written the users_matrics'+'\n')
    
    '''
    电影推荐
    '''
    
    movie_matrics = pickle.load(open('movie_matrics.p', mode='rb'))
    #recommend_same_type_movie(2)
    users_matrics = pickle.load(open('users_matrics.p', mode='rb'))
    #recommend_your_favorite_movie(30)

