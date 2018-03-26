
# coding: utf-8

# In[1]:


import json
from collections import Counter, OrderedDict
import os
from operator import itemgetter    
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from TFIDF import TFIDF
# from candidates_of_keyword import get_raw_pl , get_tfidf


# ## get tf-idf value and pl data

# In[2]:


# # pl_cnt = json.load(open('../new_Steeve_data/filter_CareerBuilder/tfidf_score_cb_filter.json','r'))
# # PATH = '../new_Steeve_data/filter_CareerBuilder/can/'
# field_names = []
# pl_cnt = get_tfidf()


# In[3]:


from pyfasttext import FastText
model = FastText('/home/vincent/atos/wiki.en.bin')


# In[31]:
def get_pl_v(NUM_PL):
    m = []
    aaa = {}
    
    # compare four fields tf-idf value
    N_SCORE = .0
    MAX_SCORE = .0
    PREDICT_FIELD = '' 
    
    for cnt_f in pl_cnt:
        for pl in j:
            if pl_cnt[cnt_f].get(pl) == None:
                continue
                N_SCORE += pl_cnt[cnt_f].get(pl)
            if N_SCORE > MAX_SCORE:
                PREDICT_FIELD = cnt_f
                MAX_SCORE = N_SCORE
            elif PREDICT_FIELD == '':
                PREDICT_FIELD = cnt_f
                        
    for pl in j:
        print(type(pl))
        if pl_cnt[PREDICT_FIELD].get(pl) == None:
            pass
        else:
            aaa[pl] = pl_cnt[PREDICT_FIELD].get(pl)
            
    for i, g in sorted(dict(aaa).items(), key=itemgetter(1), reverse=True)[:NUM_PL]:
        try:
            x = list(model[i])
            m = m + x
        except:
            pass
                
                
    if len(m) < D_WORD*NUM_PL:
        if (len(m)/D_WORD)%2 == 1:
            m = m+m[0:D_WORD]
            if len(m) == D_WORD*6:
                m = m + m[0:D_WORD*2]
            elif len(m) == D_WORD*4:
                m = m + m
            elif len(m) == D_WORD*2:
                m = m + m[0:D_WORD]
                m = m + m
            if len(m) != D_WORD*NUM_PL:
                pass
    return m

def pl_preprocessing(total_pl,NUM_PL):
    train_data = []
    train_y = []
    
    # label
    l = 0
    
    for field in total_pl:
#         print(field)
        for num, j in enumerate(field):
#             m = []
#             aaa = {}
            
#             # compare four fields tf-idf value
#             N_SCORE = .0
#             MAX_SCORE = .0
#             PREDICT_FIELD = '' 
            
            
#             for cnt_f in pl_cnt:
#                 for pl in j:
#                     if pl_cnt[cnt_f].get(pl) == None:
#                         continue
#                     N_SCORE += pl_cnt[cnt_f].get(pl)
#                 if N_SCORE > MAX_SCORE:
#                     PREDICT_FIELD = cnt_f
#                     MAX_SCORE = N_SCORE
#                 elif PREDICT_FIELD == '':
#                     PREDICT_FIELD = cnt_f
                        
#             for pl in j:
#                 print(type(pl))
#                 if pl_cnt[PREDICT_FIELD].get(pl) == None:
#                     pass
#                 else:
#                     aaa[pl] = pl_cnt[PREDICT_FIELD].get(pl)
            
#             for i, g in sorted(dict(aaa).items(), key=itemgetter(1), reverse=True)[:NUM_PL]:
#                 try:
#                     x = list(model[i])
#                     m = m + x
#                 except:
#                     pass
                
                
#             if len(m) < D_WORD*NUM_PL:
#                 if (len(m)/D_WORD)%2 == 1:
#                     m = m+m[0:D_WORD]
#                 if len(m) == D_WORD*6:
#                     m = m + m[0:D_WORD*2]
#                 elif len(m) == D_WORD*4:
#                     m = m + m
#                 elif len(m) == D_WORD*2:
#                     m = m + m[0:D_WORD]
#                     m = m + m
#                 if len(m) != D_WORD*NUM_PL:
#                     continue
                            
            train_data.append(get_pl_v(NUM_PL))
            train_y.append(l)
        l += 1
#                             print(i)
#     print(t,s)
            
    return train_data,train_y


# In[32]:


### 額外程式，主要去打亂合併的資料，並做k-fold 的取資料
class CrossValidationFolds(object):
    
    def __init__(self, data, labels, num_folds, shuffle=True):
        self.data = data
        self.labels = labels
        self.num_folds = num_folds
        self.current_fold = 0
        
        # Shuffle Dataset
        if shuffle:
            perm = np.random.permutation(self.data.shape[0]) ##隨機打亂資料
            data = data[perm]
            labels = labels[perm]
    
    def split(self):
        current = self.current_fold
        size = int(self.data.shape[0]/self.num_folds) # 30596 / 5 一塊k的size大小
        
        index = np.arange(self.data.shape[0]) 

        # 利用 True/False 抓出 validation 區塊
        lower_bound = index >= current*size # validation 下界
        upper_bound = index < (current + 1)*size # 上界

        cv_region = lower_bound*upper_bound

        cv_data = self.data[cv_region] # 利用 True/False 抓出 True 的資料
        train_data = self.data[~cv_region]
        
        cv_labels = self.labels[cv_region]
        train_labels = self.labels[~cv_region]
        
        self.current_fold += 1 ## 丟回下一的fold
        return (train_data, train_labels), (cv_data, cv_labels)


# In[1]:


# FOLDS = 5
# total_pl = get_raw_pl()
# x, y = pl_preprocessing(total_pl)
# # x2, y2 = load_pl('../new_Steeve_data/filter_Dice/can/')
# # x += x2
# # y += y2
# x= np.array(x)
# y = np.array(y)

# ###### test 同 training data #######
# X_train, X_test1, Y_train, y_test1 = train_test_split(x, y, test_size = 0.2)
# data = CrossValidationFolds(X_train, Y_train, FOLDS)
# (X_train1, y_train1), (X_valid1, y_valid1) = data.split()

# ###### test 不同 training data #######
# # data = CrossValidationFolds(x, y, FOLDS)
# # (X_train1, y_train1), (X_valid1, y_valid1) = data.split()

# # X_test1,y_test1 = load_pl('../new_Steeve_data/filter_Dice/can/')
# # X_test1 = np.array(X_test1)
# # y_test1 = np.array(y_test1)

# ##### testing data ######
# # Tx = X_test1[0]
# # Ty = y_test1[0]
# # Tx = Tx.reshape([1,-1])
# # print(Tx.shape)
# # print(X_test1.shape)


# In[ ]:


def L_layers_model(X, h_units, n_class, dropout=0.5):
    # default he_init: factor=2.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32
    he_init = tf.contrib.layers.variance_scaling_initializer()
    keep_prob = tf.placeholder(tf.float32)
    
    with tf.name_scope("DNN"):
        hidden1 = tf.layers.dense(X, 128, activation=tf.nn.relu, name="hidden1",use_bias=True, kernel_initializer= he_init,
                                      bias_initializer=he_init)
        dropout1= tf.layers.dropout(hidden1, rate=0.5,name="dropout1")
        
        hidden2 = tf.layers.dense(dropout1, 128, activation=tf.nn.relu, name="hidden2",use_bias=True, kernel_initializer= he_init,
                                      bias_initializer=he_init)
        dropout2= tf.layers.dropout(hidden2, rate=0.5,name="dropout2")
        
        hidden3 = tf.layers.dense(dropout2, 128, activation=tf.nn.relu, name="hidden3",use_bias=True, kernel_initializer= he_init,
                                      bias_initializer=he_init)
        dropout3= tf.layers.dropout(hidden3, rate=0.5,name="dropout3")
        
        hidden4 = tf.layers.dense(dropout3, 128, activation=tf.nn.relu, name="hidden4",use_bias=True, kernel_initializer= he_init,
                                      bias_initializer=he_init)
        dropout4= tf.layers.dropout(hidden4, rate=0.5,name="dropout4")
        
        hidden5 = tf.layers.dense(dropout4, 128, activation=tf.nn.relu, name="hidden5",use_bias=True, kernel_initializer= he_init,
                                      bias_initializer=he_init)
        dropout5= tf.layers.dropout(hidden5, rate=0.5,name="dropout5")
            
        # 結合之後的 tf.nn.sparse_softmax_cross_entropy_with_logits [128 , 5]
        logits = tf.layers.dense(dropout5, n_class, name="logits")
    
    return logits

def train_op(y, logits):
    with tf.name_scope("calc_loss"):
        entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(entropy, name="loss")
    
    ## 此區使用AdamOptimizer 的優化器進行梯度優化
    with tf.name_scope("train"):
        batch = tf.Variable(0)

        learning_rate = tf.train.exponential_decay(
                1e-4,  # Base learning rate.
                batch * batch_size,  # Current index into the dataset.
                n_train,  # Decay step.
                0.95,  # Decay rate.
                staircase=True)
        
        optimizer = tf.train.AdamOptimizer(0.001)
        training_op = optimizer.minimize(loss, global_step=batch,name="training_op")

    return (loss, training_op)

def acc_model(y, logits):
    #計算正確率   
    with tf.name_scope('calc_accuracy'):
        correct = tf.equal(tf.argmax(logits, 1), y)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32),name="accuracy")

    #計算precision 因返回會有兩個值，只取後者
    with tf.name_scope("precision"):
        _, precision = tf.metrics.precision(predictions = tf.argmax(logits,1), labels=y)

    #計算recall 因返回會有兩個值，只取後者
    with tf.name_scope('recall'):
        _, recall = tf.metrics.recall(predictions = tf.argmax(logits,1), labels=y)

    return (accuracy, precision, recall)


def shuffle_data(data, labels):
    idx = np.random.permutation(len(data))
    data, label = data[idx], labels[idx]
    return (data, label)


# In[ ]:


# ### 先前設置 
# FOLDS = 5
# in_units = D_WORD*NUM_PL
# n_class = 6 # 題目要求只要辨識 0 ,1 ,2 ,3 及4 ，共5個類別

# n_train = len(X_train1) # train資料的長度
# batch_size = 50
# n_batch = n_train // batch_size

# X = tf.placeholder(tf.float32,[None,in_units],name="X") # 初始化x資料型態為[None,784]
# y = tf.placeholder(tf.int64, shape=(None), name="y") # 初始化y資料型態[None]


# In[ ]:


# logits = L_layers_model(X, 128, n_class, 0.5)
# Y_proba=tf.nn.softmax(logits,name="Y_proba")
# loss, train_op = train_op(y, logits)
# accuracy, precision, recall = acc_model(y, logits)

# prediction=tf.argmax(Y_proba,1)

# saver = tf.train.Saver()  # call save function
# config = tf.ConfigProto(device_count = {'GPU': 1}) #指定gpu


# In[ ]:


# # Params for Train
# epochs = 1000 # 10 for augmented training data, 20 for training data
# val_step = 100 # 當 50 步時去算一次驗證資料的正確率
# # display_step = 100  # 你可以用這個去建一個規則去看train的正確率，但本專案先不用

# # Training cycle
# max_acc = 0. # Save the maximum accuracy value for validation data
# early_stop_limit = 0 # 紀錄early_stop的值

# init = tf.global_variables_initializer()
# init_l = tf.local_variables_initializer()


# In[4]:


def run(sess, X_train1, y_train1, X_valid1, y_valid1):
    early_stop_limit = 0
    max_acc = 0.
        
    sess.run([init, init_l])
    summary_writer = tf.summary.FileWriter('tensorbord/', graph=tf.get_default_graph())
    # 可以自己選擇大概要幾個epoch，因為我們有early stop的方法，所以不怕一直train
    for epoch in range(epochs):
        if early_stop_limit >= 200: 
            print('early_stop...........')
            break

        # Random shuffling
        train_data, train_label = shuffle_data(X_train1, y_train1)

        # 用批次的方式去訓練 model
        for i in range(n_batch):
            # Compute the offset of the current minibatch in the data.
            offset = (i * batch_size) % (n_train)
            batch_xs = train_data[offset:(offset + batch_size), :]
            batch_ys = train_label[offset:(offset + batch_size)]
            sess.run([train_op, loss], feed_dict={X:batch_xs, y: batch_ys})

            # 每 n step時，model去看此時的驗證資料的正確率並印出來
            if i % val_step == 0:
                val_acc = sess.run(accuracy, feed_dict={X: X_valid1, y: y_valid1})
                print("Epoch:", '%04d,' % (epoch + 1), 
                      "batch_index %4d/%4d , validation accuracy %.5f" % (i, n_batch, val_acc))

                # 透過最大驗證正確率大於每一次的驗證正確率的條件來設定 early stop
                if max_acc >= val_acc:
                    early_stop_limit += 1
                    if early_stop_limit == 200: # 自己可以去限制最大驗證正確率不再變n次時，就停止訓練
                        break

                # 如果 val_acc 大於 max_acc，則取代它並儲存一次結果
                else: # validation_accuracy > max_acc
                    early_stop_limit = 0
                    max_acc = val_acc
                    saver.save(sess, '../dnn_model.ckpt')                        
                    print('dnn_model.ckpt-' + 'complete-%04d-' % (epoch + 1) + 
                          "batch_index-%d" % i)


# In[ ]:


# with tf.Session(config=config) as sess:
#     run(sess, X_train1, y_train1, X_valid1, y_valid1)
    
#     sess.run(init_l)
#     saver.restore(sess, '../dnn_model.ckpt') # 開啟剛剛 early_stop 的 model

#     print('Acc_test :' , sess.run(accuracy, feed_dict={X: X_test1, y: y_test1}))
#     print('Prec_value :' , sess.run(precision, feed_dict={X: X_test1, y: y_test1}))
#     print('Recall_value :' , sess.run(recall, feed_dict={X: X_test1, y: y_test1}))

#     ######## Predict label #########
# #     print("predictions", prediction.eval(feed_dict={X: Tx}, session=sess),Ty)
# #     pre = sess.run(tf.argmax(logits, 1), feed_dict={X: X_test1})
# #     print(classification_report(y_test1.ravel(), pre))


# In[2]:


def get_Dnn_model(total_pl):
    
    NUM_PL = 8
    D_WORD = 300
    tf_idf = TFIDF(total_pl)
    pl_cnt, words = tf_idf.get_tfidf()
    
    x, y = pl_preprocessing(total_pl,NUM_PL)
    x = np.array(x)
    y = np.array(y)

    ###### test 同 training data #######
    X_train, X_test1, Y_train, y_test1 = train_test_split(x, y, test_size = 0.2)
    data = CrossValidationFolds(X_train, Y_train, FOLDS)
    (X_train1, y_train1), (X_valid1, y_valid1) = data.split()

    ###### test 不同 training data #######
    # data = CrossValidationFolds(x, y, FOLDS)
    # (X_train1, y_train1), (X_valid1, y_valid1) = data.split()

    # X_test1,y_test1 = load_pl('../new_Steeve_data/filter_Dice/can/')
    # X_test1 = np.array(X_test1)
    # y_test1 = np.array(y_test1)

    ##### testing data ######
    # Tx = X_test1[0]
    # Ty = y_test1[0]
    # Tx = Tx.reshape([1,-1])
    # print(Tx.shape)
    # print(X_test1.shape)
    
    ### 先前設置 
    FOLDS = 5
    in_units = D_WORD*NUM_PL
    n_class = 6 # 題目要求只要辨識 0 ,1 ,2 ,3 及4 ，共5個類別

    n_train = len(X_train1) # train資料的長度
    batch_size = 50
    n_batch = n_train // batch_size

    X = tf.placeholder(tf.float32,[None,in_units],name="X") # 初始化x資料型態為[None,784]
    y = tf.placeholder(tf.int64, shape=(None), name="y") # 初始化y資料型態[None]
    
    logits = L_layers_model(X, 128, n_class, 0.5)
    Y_proba=tf.nn.softmax(logits,name="Y_proba")
    loss, train_op = train_op(y, logits)
    accuracy, precision, recall = acc_model(y, logits)

    prediction=tf.argmax(Y_proba,1)

    saver = tf.train.Saver()  # call save function
    config = tf.ConfigProto(device_count = {'GPU': 1}) #指定gpu
    
    # Params for Train
    epochs = 1000 # 10 for augmented training data, 20 for training data
    val_step = 100 # 當 50 步時去算一次驗證資料的正確率

    # Training cycle
    max_acc = 0. # Save the maximum accuracy value for validation data
    early_stop_limit = 0 # 紀錄early_stop的值

    init = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    
    with tf.Session(config=config) as sess:
        run(sess, X_train1, y_train1, X_valid1, y_valid1)
        sess.run(init_l)
        saver.restore(sess, '../dnn_model.ckpt') # 開啟剛剛 early_stop 的 model

        print('Acc_test :' , sess.run(accuracy, feed_dict={X: X_test1, y: y_test1}))
        print('Prec_value :' , sess.run(precision, feed_dict={X: X_test1, y: y_test1}))
        print('Recall_value :' , sess.run(recall, feed_dict={X: X_test1, y: y_test1}))
    

    ######## Predict label #########
#     print("predictions", prediction.eval(feed_dict={X: Tx}, session=sess),Ty)
#     pre = sess.run(tf.argmax(logits, 1), feed_dict={X: X_test1})
#     print(classification_report(y_test1.ravel(), pre))


# In[3]:


def predict_field(cv):
    
    ####### Predict label #########
    print("predictions", prediction.eval(feed_dict={X: Tx}, session=sess),Ty)
#     pre = sess.run(tf.argmax(logits, 1), feed_dict={X: X_test1})
#     print(classification_report(y_test1.ravel(), pre))
    
    return field

