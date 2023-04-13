import tensorflow as tf
class Layer1(tf.keras.layers.Layer):
    def __init__(self,num_units,d = 10,k = 0.3,b = 0.3,name = 'lb'):
        super(Layer1,self).__init__()
        self.weight = self.add_weight(name= 'weight2',shape = (2 * num_units,num_units),initializer = tf.keras.initializers.GlorotNormal(),regularizer='l2',trainable=True)
        self.bias = self.add_weight(name = 'bias',shape= (1,num_units),initializer = tf.keras.initializers.GlorotNormal(),regularizer='l2',trainable= True)
        self.d = d
        self.k = k
        self.b = b
    @tf.function
    def call(self,factor,interact_emb,h,training = None):
        k = self.k
        d = self.d
        b = self.b

        gate = k + (1 - k) / (1 + tf.exp(-d * (factor - b)))
        w = tf.matmul(tf.concat([h,interact_emb],-1),self.weight)  + self.bias
        w = tf.nn.sigmoid(w * gate)
        
        return w
        
