from sklearn import metrics
from sklearn.metrics import mean_squared_error

import tensorflow as tf
import numpy as np
from model import *
from data_helper import *
import math
import json
import argparse
from tqdm import tqdm
def compute_auc(all_target, all_pred):
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)

def binary_entropy(target, pred):
    loss = target * np.log(np.maximum(1e-10, pred)) + (1.0 - target) * np.log(np.maximum(1e-10, 1.0 - pred))
    return np.average(loss) * -1.0

def train_one_epoch(recurrent,optimizer,batch_size,Topics_all, Resps_all,Masks_all,time_factor_all,attempts_factor_all,hints_factor_all):
    all_pred = []
    all_target = []
    bce = tf.keras.losses.BinaryCrossentropy(name = 'bce',reduction='sum')
    n = len(Topics_all) // batch_size
    shuffled_ind = np.arange(len(Topics_all))
    np.random.shuffle(shuffled_ind)
    Topics_all = Topics_all[shuffled_ind]
    Resps_all = Resps_all[shuffled_ind]
    Masks_all = Masks_all[shuffled_ind]
    time_factor_all = time_factor_all[shuffled_ind]
    attempts_factor_all = attempts_factor_all[shuffled_ind]
    hints_factor_all = hints_factor_all[shuffled_ind]

    for idx in tqdm(range(n)):
        Topics = Topics_all[idx * batch_size : (idx + 1) * batch_size,:]
        Resps = Resps_all[idx * batch_size : (idx + 1) * batch_size, :]
        Masks = Masks_all[idx * batch_size : (idx + 1) * batch_size, :]
        time_factor = time_factor_all[idx * batch_size : (idx + 1) * batch_size, :]
        attempts_factor = attempts_factor_all[idx * batch_size : (idx + 1) * batch_size, :]
        hints_factor = hints_factor_all[idx * batch_size : (idx + 1) * batch_size, :]
        with tf.GradientTape() as tape:
            y_pred,_ = recurrent(Topics,Resps, time_factor,attempts_factor,hints_factor,Masks, training = True)
            y_pred = tf.reshape(y_pred,[-1,1])
            
            Resps_q = Resps[:,1:]
            y_true = tf.reshape(Resps_q, [-1,1])

            Masks_q = Masks[:,1:]
            Masks_q = tf.reshape(Masks_q,[-1])
            valid_num = tf.cast(tf.reduce_sum(Masks_q),tf.float32)
        
            total_loss = bce(y_true,y_pred,sample_weight = Masks_q)
            batch_loss = tf.divide(total_loss , valid_num)

            gradients = tape.gradient(batch_loss, recurrent.trainable_variables)
            optimizer.apply_gradients(zip(gradients,recurrent.trainable_variables))
            
        all_pred.append(tf.gather_nd(tf.reshape(y_pred,[-1]),tf.where(tf.equal(Masks_q,1))))
        all_target.append(tf.gather_nd(tf.reshape(y_true,[-1]),tf.where(tf.equal(Masks_q,1))))
    all_pred = np.concatenate(all_pred, axis=0)
    all_target = np.concatenate(all_target, axis=0)
    print(all_pred)
    loss = binary_entropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    acc = compute_accuracy(all_target, all_pred)
    return loss,auc,acc
def test_one_epoch(recurrent,batch_size,Topics_all, Resps_all,Masks_all,time_factor_all,attempts_factor_all,hints_factor_all):
    all_pred,all_target = [],[]
    n = len(Topics_all) // batch_size
    for idx in range(n):
        Topics = Topics_all[idx * batch_size : (idx + 1) * batch_size,:]
        Resps = Resps_all[idx * batch_size : (idx + 1) * batch_size, :]
        Masks = Masks_all[idx * batch_size : (idx + 1) * batch_size, :]
        time_factor = time_factor_all[idx * batch_size : (idx + 1) * batch_size, :]
        attempts_factor = attempts_factor_all[idx * batch_size : (idx + 1) * batch_size, :]
        hints_factor = hints_factor_all[idx * batch_size : (idx + 1) * batch_size, :]

        y_pred,improve = recurrent(Topics, Resps, time_factor,attempts_factor,hints_factor,Masks, training = False)
        y_pred = tf.reshape(y_pred,[-1])
        
        Resps_q = Resps[:,1:]
        y_true = tf.reshape(Resps_q, [-1])

        Masks_q = Masks[:,1:]
        Masks_q = tf.reshape(Masks_q,[-1])

        all_pred.append(tf.gather_nd(y_pred,tf.where(tf.equal(Masks_q,1))))
        all_target.append(tf.gather_nd(y_true,tf.where(tf.equal(Masks_q,1))))
    all_pred = np.concatenate(all_pred, axis=0)
    all_target = np.concatenate(all_target, axis=0)
    
    loss = binary_entropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    rmse = mean_squared_error(all_target, all_pred,squared = False)
    acc = compute_accuracy(all_target, all_pred)
    
    return loss,auc,acc,rmse
        
class LBKT():
    def __init__(self,num_topics,dim_tp, num_resps,num_units, dropout,dim_hidden,memory_size,BATCH_SIZE,q_matrix):
        super(LBKT,self).__init__()
        self.recurrent = Recurrent(num_topics,dim_tp, num_resps,num_units, dropout,dim_hidden,memory_size,BATCH_SIZE,q_matrix)
        self.batch_size = BATCH_SIZE
    def train(self, train_data, test_data, epoch: int, lr,steps_per_epoch,save_path) -> ...:
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                        lr,
                        decay_steps = steps_per_epoch,
                        decay_rate = 0.5
                        )
        optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)

        
        patience = 5
        count = 0
        min_delta = 0.000001
        best_test_auc = 0
        for idx in range(epoch):
            train_loss, train_auc, train_acc = train_one_epoch(self.recurrent,  optimizer,self.batch_size,*train_data)
            print("[Epoch %d] LogisticLoss: %.6f" % (idx, train_loss))

            if test_data is not None:
                valid_loss, valid_auc, valid_acc,valid_rmse = self.eval(test_data)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f" % (idx, valid_auc, valid_acc,valid_rmse))
                if valid_auc > best_test_auc:
                    best_test_auc = valid_auc
                    self.save(save_path,optimizer)
                if valid_auc - best_test_auc < min_delta:
                    count += 1
                    if count >= patience:
                        print('early stopping')
                        break
                else:
                    count = 0

    def eval(self, test_data) -> ...:
        
        return test_one_epoch(self.recurrent, self.batch_size,*test_data)

    def save(self, filepath,optimizer) -> ...:
        ckpt = tf.train.Checkpoint(recurrent = self.recurrent, optimizer = optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt,filepath, max_to_keep = 1)
        ckpt_manager.save()
    def load(self, filepath) -> ...:
        ckpt = tf.train.Checkpoint(recurrent = self.recurrent)
        ckpt.restore(filepath).expect_partial()

task = 'LBKT_assist2009'
print('Task:',task)
data_path = '../data2/'

train_data_path = data_path + 'new_train_data.json'
valid_data_path = data_path + 'new_valid_data.json'
test_data_path = data_path + 'new_test_data.json'
topicIndex_path = data_path + 'topic2index.json'
kcIndex_path = data_path + 'kc2index.json'
topic_time_para_path = data_path + 'topic_time_para.json'
time_factor_path = data_path + 'time_factor.json'
attempts_factor_path = data_path + 'attempts_factor.json'
hints_factor_path = data_path + '/hints_factor.json'
checkpoint_path = data_path + 'checkpoints/' + task
q_matrix_path = data_path + 'q_matrix.json'
with open(topicIndex_path,'r',encoding='utf-8') as f:
    topicIndex = json.loads(f.read())
with open(kcIndex_path,'r',encoding='utf-8') as f:
    kc2index = json.loads(f.read())
print(len(topicIndex))
with open(q_matrix_path,'r',encoding='utf-8') as f:
    q_matrix = json.loads(f.read())
q_matrix = tf.reshape(q_matrix,[len(topicIndex),len(kc2index)])
q_matrix = tf.cast(q_matrix,tf.float32)
parser = argparse.ArgumentParser(description='add parameter')
parser.add_argument('-bs','--batch_size',default = 16)
parser.add_argument('-lr','--lrate',default = 0.005)
parser.add_argument('-qf','--q_factor',default = 0.01)
args = parser.parse_args()

tf.random.set_seed(2023)

pad_len = 100
dim_tp = 128
num_resps = 2
num_units = 128
dropout  = 0.2 #0.2 kt529
dim_hidden = 50
EPOCHS = 100
BUFFER_SIZE = 1000
BATCH_SIZE = int(args.batch_size)
lr = float(args.lrate)#0.005 #0.001
q_factor = float(args.q_factor)



train_Topics,train_Resps, train_AnswerTime,train_Attempts,train_Hints, train_Masks,train_time_factor,train_attempts_factor,train_hints_factor = form_data(train_data_path,pad_len, topicIndex_path,time_factor_path,attempts_factor_path,hints_factor_path)
valid_Topics,valid_Resps, valid_AnswerTime,valid_Attempts,valid_Hints, valid_Masks , valid_time_factor,valid_attempts_factor,valid_hints_factor = form_data(valid_data_path,pad_len, topicIndex_path,time_factor_path,attempts_factor_path,hints_factor_path)

if len(train_Topics) % BATCH_SIZE:
    train_Topics, train_Resps, train_AnswerTime,train_Attempts,train_Hints, train_Masks,train_time_factor,train_attempts_factor,train_hints_factor = \
        fit_batch(train_Topics, train_Resps, train_AnswerTime,train_Attempts,train_Hints, train_Masks,train_time_factor,train_attempts_factor,train_hints_factor,BATCH_SIZE,pad_len)
train_data = (train_Topics,train_Resps,train_Masks,train_time_factor,train_attempts_factor,train_hints_factor)
if len(valid_Topics) % BATCH_SIZE:
    valid_Topics, valid_Resps, valid_AnswerTime,valid_Attempts,valid_Hints, valid_Masks , valid_time_factor,valid_attempts_factor,valid_hints_factor = \
        fit_batch(valid_Topics,valid_Resps, valid_AnswerTime,valid_Attempts,valid_Hints, valid_Masks , valid_time_factor,valid_attempts_factor,valid_hints_factor,BATCH_SIZE,pad_len)

valid_data = (valid_Topics,valid_Resps,valid_Masks, valid_time_factor,valid_attempts_factor,valid_hints_factor)

test_Topics, test_Resps, test_AnswerTime,test_Attempts,test_Hints, test_Masks,test_time_factor,test_attempts_factor,test_hints_factor = form_data(test_data_path,pad_len, topicIndex_path,time_factor_path,attempts_factor_path,hints_factor_path)


if len(test_Topics) % BATCH_SIZE:
    test_Topics, test_Resps, test_AnswerTime,test_Attempts,test_Hints, test_Masks,test_time_factor,test_attempts_factor,test_hints_factor = \
        fit_batch(test_Topics, test_Resps, test_AnswerTime,test_Attempts,test_Hints, test_Masks,test_time_factor,test_attempts_factor,test_hints_factor,BATCH_SIZE,pad_len)
test_data = (test_Topics, test_Resps, test_Masks,test_time_factor,test_attempts_factor,test_hints_factor)
num_topics = len(topicIndex)

memory_size = len(kc2index)

steps_per_epoch = int((len(train_Topics) + BATCH_SIZE - 1)/ BATCH_SIZE)
print('steps_per_epoch ',steps_per_epoch)
best_test_auc = 0
q_matrix = (1 - q_factor) * q_matrix + q_factor * np.ones_like(q_matrix)

model = LBKT(num_topics,dim_tp, num_resps,num_units, dropout,dim_hidden,memory_size,BATCH_SIZE,q_matrix)
model.train(train_data,valid_data,EPOCHS,lr,steps_per_epoch,checkpoint_path)
model.load(checkpoint_path + '/ckpt-1')
loss,auc,acc,rmse = model.eval(test_data)
print('loss:{:.6f},auc:{:.6f}, acc:{:.6f}, rmse:{:.6f}'.format(loss,auc,acc,rmse))


