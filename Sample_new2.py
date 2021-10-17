from datetime import datetime
import math
import time
import pandas as pd
import pickle
import argparse
import tensorflow as tf
import numpy as np
import dataset_info
import model_info
import os
model_name = 'InceptionResNetV2'
datasetsize=128
batch_size = 256 #batch size

epoch_start = 0
epoch_end = 0
dataset = dataset_info.select_dataset(datasetsize)
num_classes = dataset['num_classes']
img_rows = dataset['img_rows']
img_cols = dataset['img_cols']
img_channels = dataset['img_channels']
num_data = dataset['num_data']
num_test = dataset['num_test']

prof_point = 15 #prof_point
batch_num = math.ceil(num_data/batch_size)
epochs = math.ceil(prof_point)
prof_start = math.floor(batch_num * prof_point)
prof_len = 1 #prof_len
prof_range = '{}, {}'.format(prof_start, prof_start + prof_len)
optimizer = 'SGD'

###################### Build Fake Dataset ######################
x_train_shape = (num_data, img_rows, img_cols, img_channels)
y_train_shape = (num_data, 1)

x_test_shape = (num_test, img_rows, img_cols, img_channels)
y_test_shape = (num_test, 1)

x_train = np.random.rand(*x_train_shape)
y_train = np.random.randint(num_classes, size=y_train_shape)
x_test = np.random.rand(*x_test_shape)
y_test = np.random.randint(num_classes, size=y_test_shape)
###############################################################
if tf.keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], img_channels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], img_channels, img_rows, img_cols)
    input_shape = (img_channels, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)
    input_shape = (img_rows, img_cols, img_channels)
    
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Select model from model info module
model = model_info.select_model(model_name, input_shape, num_classes)

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])


class BatchTimeCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.all_times = []

    def on_train_end(self, logs=None):
        time_filename = "sample2.pickle"
        time_file = open(time_filename, 'ab')
        pickle.dump(self.all_times, time_file)
        time_file.close()

    def on_epoch_begin(self, epoch, logs=None):
        print(epoch)
        global epoch_start
        self.epoch_times = []
        self.epoch_time_start = time.time()
        #print(datetime.fromtimestamp(self.epoch_time_start))
        epoch_start=datetime.fromtimestamp(self.epoch_time_start).strftime('%Y/%m/%d %H:%M:%S')
        print(epoch_start)
    def on_epoch_end(self, epoch, logs=None):
        global epoch_end
        self.epoch_time_end = time.time()
        #self.all_times.append(self.epoch_time_end - self.epoch_time_start)
        self.all_times.append(self.epoch_times)
        #print(datetime.fromtimestamp(self.epoch_time_end))
        #print(datetime.fromtimestamp(self.epoch_time_end).strftime('%Y/%m/%d %H:%M:%S.%f')[:-3])
        epoch_end=datetime.fromtimestamp(self.epoch_time_end).strftime('%Y/%m/%d %H:%M:%S')
        print(epoch_end)
        time.sleep(20)
        #------- 여기서부터 nvidia-smi 데이터 epoch 별로 잘라내는 부분~ ----------------------------------------------------------------------
        filename= './Data.csv'
        Log_Data = pd.read_csv(filename)
        Log_Data["timestamp"]=Log_Data["timestamp"][:].str[:19]   # 초위에 소수점3자리 잘라냄
        start_index = (Log_Data[Log_Data["timestamp"][:].str[:19] == epoch_start].index[0]) # strart 지점이랑 같은 인덱스 찾기
        end_index = (Log_Data[Log_Data["timestamp"][:].str[:19] == epoch_end].index[0]) # end 지점이랑 가틍 인덱스 찾기
        Log_Data = Log_Data[start_index:end_index+1]
        epoch_ver_filename= './'+str(model_name)+'_batch_size'+str(batch_size)+'_datasize'+str(datasetsize)+'_epoch'+str(epoch+1)+'.csv'
        Log_Data.to_csv(epoch_ver_filename, index=False, encoding='cp949')
        #-----------------------------------------------------------------------------------------       
    def on_train_batch_begin(self, batch, logs=None):
        self.batch_time_start = time.time()
    def on_train_batch_end(self, batch, logs=None):
        self.epoch_times.append(time.time() - self.batch_time_start)
latency_callback = BatchTimeCallback()

model.fit(x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test),
    callbacks = [latency_callback])
