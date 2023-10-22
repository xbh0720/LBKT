import numpy as np
import json
import scipy.stats
import math

def fit_batch(Topics, Resps, AnswerTime,Attempts,Hints, Masks,Time_Factor,Attempts_Factor,Hints_Factor,batch_size,pad_len):
    left = len(Topics) % batch_size
    padding = batch_size - left
    for i in range(padding):
        pads = np.array([0 for j in range(pad_len)])
        Topics = np.r_[Topics,[pads]]
        
        Resps = np.r_[Resps,[pads]]
        AnswerTime = np.r_[AnswerTime,[pads]]
        Attempts = np.r_[Attempts,[pads]]
        Hints = np.r_[Hints,[pads]]
        Masks = np.r_[Masks,[pads]]
        Time_Factor = np.r_[Time_Factor,[pads]]
        Attempts_Factor = np.r_[Attempts_Factor,[pads]]
        Hints_Factor = np.r_[Hints_Factor,[pads]]
    return Topics,Resps, AnswerTime,Attempts,Hints, Masks,Time_Factor,Attempts_Factor,Hints_Factor
def form_data(input_data,pad_len,topic2index_path,topic_factor_path,attemtps_factor_path,hints_factor_path):
    with open(input_data,'r',encoding='utf-8') as f1,\
        open(topic2index_path,'r',encoding='utf-8') as f2,\
        open(topic_factor_path,'r',encoding='utf-8') as f3,\
        open(attemtps_factor_path,'r',encoding='utf-8') as f4,\
        open(hints_factor_path,'r',encoding='utf-8') as f5:
        topic2index = json.loads(f2.read())
        
        time_factor_dic = json.loads(f3.read())
        attempts_factor_dic = json.loads(f4.read())
        hints_factor_dic = json.loads(f5.read())
        print(len(topic2index))
        
        Topics,Resps,AnswerTime,Attempts,Hints,Masks = [],[],[],[],[],[]
        
        Time_Factor = []
        Attempts_Factor = []
        Hints_Factor = []
        records = json.loads(f1.read())
        for user in records.keys():
            seq = records[user]
            
            topics = [topic2index[str(i[0])] for i in seq]
            
            resps = [i[2] for i in seq]
            atime = [i[4] for i in seq]

            time_factor = time_factor_dic[user]      
            attempts_factor = attempts_factor_dic[user]
            hints_factor = hints_factor_dic[user]
            attempts = [i[5] for i in seq]
            hints = [i[6] for i in seq]
            n = len(seq) // pad_len
            if len(seq) > pad_len:
                for j in range(n):
                    begin = j * pad_len
                    end = (j + 1) * pad_len
                    Topics.append(topics[begin:end])  
                    Resps.append(resps[begin:end])
                    AnswerTime.append(atime[begin:end])
                    Attempts.append(attempts[begin:end])
                    Hints.append(hints[begin:end])
                    Time_Factor.append(time_factor[begin : end])
                    Attempts_Factor.append(attempts_factor[begin:end])
                    Hints_Factor.append(hints_factor[begin:end])
                    Masks.append([1] * pad_len)
            left = len(seq) % pad_len
            if left < 10:
                continue
            padding = pad_len - left

            topics_pad = topics[n*pad_len :] +[0] * padding

            resps_pad = resps[n* pad_len : ] + [0] * padding
            atime_pad = atime[n* pad_len : ] + [0] * padding
            attempts_pad = attempts[n * pad_len : ] + [0] * padding
            hints_pad = hints[n * pad_len : ] + [0] * padding
            time_factor_pad = time_factor[n * pad_len :] + [0]* padding 
            attempts_factor_pad = attempts_factor[n*pad_len : ] + [0] * padding
            hints_factor_pad = hints_factor[n*pad_len : ] + [0] *padding
            masks_pad = [1] * left +[0] * padding

            Topics.append(topics_pad)
            Resps.append(resps_pad)
            AnswerTime.append(atime_pad)
            Attempts.append(attempts_pad)
            Hints.append(hints_pad)
            Masks.append(masks_pad)
            Time_Factor.append(time_factor_pad)
            Attempts_Factor.append(attempts_factor_pad)
            Hints_Factor.append(hints_factor_pad)

        Topics = np.array(Topics, dtype = np.int32)
        Resps = np.array(Resps, dtype = np.int32)
        AnswerTime = np.array(AnswerTime)
        Attempts = np.array(Attempts, dtype=np.int32)
        Hints = np.array(Hints, dtype=np.int32)
        Time_Factor = np.array(Time_Factor)
        Attempts_Factor = np.array(Attempts_Factor)
        Hints_Factor = np.array(Hints_Factor)
        Masks  = np.array(Masks, dtype=np.int32)

    return Topics,Resps, AnswerTime,Attempts,Hints, Masks,Time_Factor,Attempts_Factor,Hints_Factor
