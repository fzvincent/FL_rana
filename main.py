import tensorflow as tf
import numpy as np

import nest_asyncio
from tempfile import TemporaryFile
nest_asyncio.apply()

from collections import *
import tensorflow_federated as tff
import random
from matplotlib import pyplot as plt
import math
import sys
import time
#import data_store
import shelve
import collections

from readEx import *
#from var import *
model='cifar10'
mnist = tf.keras.datasets.mnist
cifar10=tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
N = y_train.shape[0]
x_test_list = [x_test]
y_test_list = [y_test]

def nonIIDGen(beta=1):
    min_size = 0
    min_require_size = 10
    def record_net_data_stats(y_train, net_dataidx_map):
        net_cls_counts = {}
        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp
        #logger.info('Data statistics: %s' % str(net_cls_counts))
        return net_cls_counts

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            # logger.info('proportions1: ', proportions)
            # logger.info('sum pro1:', np.sum(proportions))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
            # logger.info('proportions2: ', proportions)
            proportions = proportions / proportions.sum()
            # logger.info('proportions3: ', proportions)
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            # logger.info('proportions4: ', proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    net_dataidx_map = {} # data map
    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    # change format of xtrain
    train=[]
    train_label=[]
    for j in range(n_parties):
        train.append([x_train[i] for i in net_dataidx_map[j]])
        train_label.append([y_train[i] for i in net_dataidx_map[j]])
    x_train_noniid=train
    y_train_noniid=train_label
    return x_train_noniid,y_train_noniid




def batch_format_fn(element):
    '''Flatten a batch `pixels` and return the features as an `OrderedDict`.'''
    if model=='cifar10':
        return collections.OrderedDict(x=element['pixels'], y=element['label'])
    elif model=='mnist':
        return collections.OrderedDict(x=tf.expand_dims(element['pixels'], -1), y=element['label'])

def preprocess(dataset,batchSize,batchCount,seednumber):
    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER, seed=seednumber).batch(
      batchSize).take(batchCount).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

def make_federated_data(client_data, client_label,tnt,seednumber=1):
    temp_list=[]
    if tnt=='train':
      batchSize,batchCount=batchSizeTrain,batchCountTrain
    elif tnt=='test':
      batchSize, batchCount = batchSizeTest, batchCountTest
    # for x in range(n_parties):
    #     dataset1=tf.data.Dataset.from_tensor_slices({'pixels':client_data[x], 'label':client_label[x]})
    #     # dataset2 = preprocess(dataset1)
    #     # temp_list.append(dataset2)
    #     temp_list.append(preprocess(dataset1,batchSize,batchCount))
    temp_list = [preprocess(
      tf.data.Dataset.from_tensor_slices({'pixels': client_data[i], 'label': client_label[i]}),
      batchSize,batchCount,seednumber)
      for i in range(len(client_data))]
    return temp_list

#print('preparing FL training data')
#federated_train_data = make_federated_data(x_train, y_train,'train',seednumber=1)



print('preparing FL testing data')
federated_test_data = make_federated_data(x_test_list,y_test_list,'test')



cpu_device = tf.config.list_logical_devices('CPU')[0]
gpu_devices = tf.config.list_logical_devices('GPU')
# tff.backends.native.set_local_execution_context(server_tf_device=cpu_device,
#     client_tf_devices=gpu_devices,
#     #clients_per_thread=1,
#     max_fanout=100)
tff.backends.native.set_local_execution_context(
    client_tf_devices=gpu_devices,
    #clients_per_thread=1,
    max_fanout=100)


def create_keras_model():
    if model=='cifar10':
        structure=tf.keras.models.Sequential([
          #tf.keras.layers.InputLayer(input_shape=[28,28,1]),   #
          tf.keras.layers.InputLayer(input_shape=[32, 32, 3]),
          tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same', activation='relu'),
          tf.keras.layers.MaxPooling2D(),
          tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='relu'),
          tf.keras.layers.MaxPooling2D(),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(units=120, activation='relu'),
          tf.keras.layers.Dense(units=84, activation='relu'),
          tf.keras.layers.Dense(units=10, activation='softmax')
      ])
    elif model=='mnist':
        structure = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=[28,28,1]),   #
            tf.keras.layers.InputLayer(input_shape=[32, 32, 3]),
            tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=120, activation='relu'),
            tf.keras.layers.Dense(units=84, activation='relu'),
            tf.keras.layers.Dense(units=10, activation='softmax')
        ])
    return structure

def model_fn():
    keras_model = create_keras_model()
    #keras_model.summary()
    return tff.learning.from_keras_model(
      keras_model,
      input_spec=federated_test_data[0].element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

iteratives=[]
for i in range(clientCount):
    iteratives.append(tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=localSGDrate*(i+1)),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1),
    use_experimental_simulation_loop=True))
iterative=iteratives[0]
state = iterative.initialize()
evaluation = tff.learning.build_federated_evaluation(model_fn,use_experimental_simulation_loop=True)


def fedAvg(state):     # fix the number of clients
    record_metrics=[]
    record_valid=[]
    record_time=[]
    for round_num in range(1, NUM_ROUNDS+1):
        selected_client_data=federated_train_data
        with tf.profiler.experimental.Profile('multigpu'):
            state, metrics = iterative.next(state, selected_client_data)
        record_metrics.append(metrics['train']['sparse_categorical_accuracy'])
        if round_num % TEST_frequency == 1:
            print(time.strftime("%Y%m%d %H%M%S"))
            metrics_valid = evaluation(state.model, federated_test_data)
            record_valid.append(metrics_valid['sparse_categorical_accuracy'])
            print('validation', metrics_valid)
    return record_valid
def fedGreen(state):     # fix the number of clients
    record_metrics=[]
    record_valid=[]
    record_time=[]
    for round_num in range(1, NUM_ROUNDS+1):
        selected_client_data = [federated_train_data[i] for i in seleGreen[round_num+1]]
        with tf.profiler.experimental.Profile('multigpu'):
            state, metrics = iterative.next(state, selected_client_data)
        record_metrics.append(metrics['train']['sparse_categorical_accuracy'])
        if round_num % TEST_frequency == 1:
            print(time.strftime("%Y%m%d %H%M%S"))
            metrics_valid = evaluation(state.model, federated_test_data)
            record_valid.append(metrics_valid['sparse_categorical_accuracy'])
            print('validation', metrics_valid)
            #record_valid[round_for_valid - 1] = metrics_valid['sparse_categorical_accuracy']
    return record_valid
def fedLearn(state):     # fix the number of clients
    record_metrics=[]
    record_valid=[]
    for round_num in range(1, NUM_ROUNDS+1):
        selected_client_data = [federated_train_data[i] for i in seleLearn[round_num+1]]
        with tf.profiler.experimental.Profile('multigpu'):
            state, metrics = iterative.next(state, selected_client_data)
        record_metrics.append(metrics['train']['sparse_categorical_accuracy'])
        if round_num % TEST_frequency == 1:
            print(time.strftime("%Y%m%d %H%M%S"))
            metrics_valid = evaluation(state.model, federated_test_data)
            record_valid.append(metrics_valid['sparse_categorical_accuracy'])
            print('validation', metrics_valid)
            #record_valid[round_for_valid - 1] = metrics_valid['sparse_categorical_accuracy']
    return record_valid




typeCount=10
# 0 for FedAvg
# 1 for Green
# 2 for Learn
validData=np.zeros([typeCount,betaCount,NUM_VALID,repetion])


## computation for fig 3 & 5
for i in range(repetion):
    clientTier, latency = tierGenerator(tierTimes[1])  # 10s
    state = iterative.initialize()
    for j in range(betaCount):
        print('fig 3&5, preparing FL training data with \u03B2='+str(betaValues[j]))
        x_train_noniid,y_train_noniid=nonIIDGen(beta=betaValues[j])
        federated_train_data = make_federated_data(x_train_noniid, y_train_noniid, 'train', seednumber=i)
        print('repetion'+str(i))
        #for k in range(tierShowMax):   # j start from 0, but +1 into the tier function
        #    print(repetion)
        #    valid5sTier[i,j,k,:]=fedAvgTier(state,k+1,i)

        #validNum [i, j, :],validNumTime[i,j,:] = fedNum(state, 5)
        validData[0, j, :, i]=fedAvg(state)

        validData[1, j, :, i] = fedGreen(state)
        validData[2, j, :, i] = fedLearn(state)

## data save

np.savez("mydata0406.npz",
        validData=validData,

        )
print(1)