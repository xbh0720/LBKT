import tensorflow as tf
import numpy as np
from cell import *
class Recurrent(tf.keras.Model):
    def __init__(self, num_topics, dim_tp, num_resps, num_units, dropout,dim_hidden,memory_size,batch_size,q_matrix):
        super(Recurrent, self).__init__()
        
        self.embedding_topic = tf.keras.layers.Embedding(num_topics, dim_tp,
                                embeddings_initializer=tf.keras.initializers.GlorotNormal(),
                                #embeddings_initializer=tf.keras.initializers.GlorotUniform(),
                                trainable = True)
        self.embedding_resps = tf.keras.layers.Embedding(num_resps, dim_hidden,
                                    embeddings_initializer=tf.keras.initializers.GlorotNormal(),
                                    #embeddings_initializer=tf.keras.initializers.GlorotUniform(),
                                    trainable = True)
        self.memory_size = memory_size
        self.num_units = num_units
        self.dim_tp = dim_tp
        self.batch_size = batch_size
        self.q_matrix = tf.reshape(q_matrix,[num_topics,memory_size])
        self.rnn = tf.keras.layers.RNN(LBKTcell(num_units,memory_size,dim_tp,dropout = dropout,name = 'lbkt'),return_sequences=True)

        self.input_layer = tf.keras.layers.Dense(num_units,tf.nn.relu, trainable = True)
        self.init_h = self.add_weight(name = 'init_h',shape = (memory_size,num_units),
                          initializer=tf.keras.initializers.GlorotNormal(),
                          #initializer=tf.keras.initializers.GlorotUniform(),
                          trainable = True)        



    @tf.function
    def call(self,topics, resps, time_factor,attempt_factor,hint_factor,masks,training):
        batch_size = tf.shape(topics)[0]
        seq_len = tf.shape(topics)[1]
        topic_emb = self.embedding_topic(topics,training = training)
        resps_emb = self.embedding_resps(resps,training = training)

        correlation_weight = tf.nn.embedding_lookup(self.q_matrix,topics)
        acts_emb = self.input_layer(tf.concat([topic_emb,resps_emb],-1),training = training) #bs * seq_len * num_units
        
        time_factor = tf.expand_dims(tf.cast(time_factor,tf.float32),-1)
        attempt_factor = tf.expand_dims(tf.cast(attempt_factor,tf.float32),-1)
        hint_factor = tf.expand_dims(tf.cast(hint_factor,tf.float32),-1)
        
        h_init = tf.tile(tf.expand_dims(self.init_h,0),(self.batch_size,1,1))
        inputs = tf.concat([acts_emb,correlation_weight,topic_emb,time_factor,attempt_factor,hint_factor],-1)
        masks_v = tf.not_equal(masks,0)
        result = self.rnn(inputs,mask = masks_v,initial_state=h_init, training = training)
        preds,improve = tf.split(result,num_or_size_splits = [1,1],axis = -1)

        return preds[:,1:],improve
