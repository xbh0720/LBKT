import tensorflow as tf
from layer import *
class LBKTcell(tf.keras.layers.Layer):
    def __init__(self,num_units,memory_size,dim_tp,dropout = 0.2,name = 'lbktcell'):
        super(LBKTcell,self).__init__()
        self.num_units = num_units
        self.memory_size = memory_size
        self.dim_tp = dim_tp
        self.r = 4  
        self.time_gain = Layer1(self.num_units,name = 'time_gain')
        self.attempt_gain = Layer1(self.num_units,name = 'attempt_gain')
        self.hint_gain = Layer1(self.num_units,name = 'hint_gain')
        self.time_weight = self.add_weight(shape = (self.r,num_units + 1,num_units),
                               initializer=tf.keras.initializers.GlorotNormal(),
                               trainable = True,name = 'time_weight')
        self.attempt_weight = self.add_weight(shape = (self.r,num_units + 1,num_units),
                                  initializer=tf.keras.initializers.GlorotNormal(),
                                  trainable = True,name = 'attempt_weight')
        self.hint_weight = self.add_weight(shape = (self.r,num_units + 1,num_units),
                               initializer=tf.keras.initializers.GlorotNormal(),
                               trainable = True,name = 'hint_weight')
        self.Wf = self.add_weight(shape = (1,self.r),
                      initializer=tf.keras.initializers.GlorotNormal(),
                      trainable = True,name = 'wf')
        self.bias = self.add_weight(shape = (1,num_units),
                        initializer=tf.keras.initializers.GlorotNormal(),
                        trainable =True,name = 'bias')

        

        self.gate3 = tf.keras.layers.Dense(num_units,tf.nn.sigmoid,trainable = True)
        self.dropout = tf.keras.layers.Dropout(rate = dropout)
        self.output_layer = tf.keras.layers.Dense(num_units,tf.nn.sigmoid,trainable = True)
        
        self.state_size = tf.TensorShape((memory_size , num_units))
        self.output_size = 2
    @tf.function
    def call(self,inputs,states,training = None):
        interact_emb,correlation_weight,topic_emb,time_factor,attempt_factor,hint_factor\
            = tf.split(inputs,num_or_size_splits = [self.num_units,self.memory_size,self.dim_tp,1,1,1],axis = -1)
        h_pre = tf.squeeze(states,0)
        h_pre_tilde = tf.squeeze(tf.matmul(tf.expand_dims(correlation_weight,1),h_pre),1) #bs *1 * memory_size , bs * memory_size * d_k
        #predict performance
        b_out = tf.concat([h_pre_tilde,topic_emb],-1)
        preds = tf.reduce_sum(self.output_layer(b_out,training = training),-1) / self.num_units
        preds = tf.expand_dims(preds,-1)
        #characterize each behavior's effect
        time_gain = self.time_gain(time_factor,interact_emb,h_pre_tilde,training = training)
        attempt_gain = self.attempt_gain(attempt_factor,interact_emb,h_pre_tilde,training = training)
        hint_gain = self.hint_gain(hint_factor,interact_emb,h_pre_tilde,training = training)
        
        #capture the dependency among different behaviors
        batch_size = tf.shape(time_gain)[0]
        pad = tf.ones_like(time_factor)
        time_gain1 = tf.concat([time_gain,pad],-1) #batch_size * num_units + 1
        attempt_gain1 = tf.concat([attempt_gain,pad],-1)
        hint_gain1 = tf.concat([hint_gain,pad],-1)
        fusion_time = tf.matmul(time_gain1,self.time_weight) #r * bs  *num_units:bs * num_units + 1 ,r * num_units + 1 *num_units
        fusion_attempt = tf.matmul(attempt_gain1,self.attempt_weight)
        fusion_hint = tf.matmul(hint_gain1,self.hint_weight)
        fusion_all = fusion_time * fusion_attempt * fusion_hint
        fusion_all = tf.squeeze(tf.matmul(self.Wf,tf.transpose(fusion_all,(1,0,2)) ),1) + self.bias
        learning_gain = tf.nn.relu(fusion_all)
            
        LG = tf.matmul(tf.expand_dims(correlation_weight,-1),tf.expand_dims(learning_gain,1)) #bs * 1 *d_k,bs * memory_size * 1
        
        #forget effect
        forget_gate = self.gate3(tf.concat([h_pre,
                      tf.tile(tf.expand_dims(interact_emb,1),(1,self.memory_size,1)),
                      tf.tile(tf.expand_dims(time_factor,1),(1,self.memory_size,50)),
                      tf.tile(tf.expand_dims(attempt_factor,1),(1,self.memory_size,50)),
                      tf.tile(tf.expand_dims(hint_factor,1),(1,self.memory_size,50))
                      ],-1))
        LG = self.dropout(LG,training = training)
        h = h_pre * forget_gate + LG

        #compute learning gain
        h_tilde = tf.squeeze(tf.matmul(tf.expand_dims(correlation_weight,1),h),1)
        new_b_out = tf.concat([h_tilde,topic_emb],-1)
        after_preds = tf.reduce_sum(self.output_layer(new_b_out,training = training),-1) / self.num_units

        after_preds = tf.expand_dims(after_preds,-1)
        improve = (after_preds - preds) / (1 - preds) #对于不认真的情况不应该有很大提升
        result = tf.concat([preds,improve],-1)
        return result,h
