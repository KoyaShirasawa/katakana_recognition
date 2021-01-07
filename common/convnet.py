#ALL Conv Batchnorm & Affine Batchnorm
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.layers import Convolution, MaxPooling, ReLU, Affine, SoftmaxWithLoss, Dropout, BatchNormalization
from common.optimizer import RMSProp

class ConvNet:
    def __init__(self, input_dim=(1, 28, 28),
                 use_conv2=True, use_affine2=True,
                 conv_param={'filter_num':128, 'filter_size':3, 'pad':1, 'stride':1},
                 pool_param={'pool_size':2, 'pad':1, 'stride':2},
                 conv_param2={'filter_num2':128, 'filter_size2':3, 'pad2':1, 'stride2':1},
                 pool_param2={'pool_size2':2, 'pad2':1, 'stride2':2},
                 hidden_size=128, hidden_size2=128, output_size=15, weight_init_std=0.01, 
                 use_batchnorm_C1=False, use_batchnorm_C2=False, use_batchnorm_A1=False, use_batchnorm_A2=False,
                 use_dropout_A1=False, dropout_ratio_A1=0.5, use_dropout_A2=False, dropout_ratio_A2=0.5, 
                 use_succession=False, data_num=1, prediction_mode=False):
        
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        
        pool_size = pool_param['pool_size']
        pool_pad = pool_param['pad']
        pool_stride = pool_param['stride']

        filter_num2 = conv_param2['filter_num2']
        filter_size2 = conv_param2['filter_size2']
        filter_pad2 = conv_param2['pad2']
        filter_stride2 = conv_param2['stride2']
        
        pool_size2 = pool_param2['pool_size2']
        pool_pad2 = pool_param2['pad2']
        pool_stride2 = pool_param2['stride2']
        
        input_size = input_dim[1]
        conv_output_size = (input_size + 2*filter_pad - filter_size) // filter_stride + 1 # 畳み込み後のサイズ(H,W共通)
        pool_output_size = (conv_output_size + 2*pool_pad - pool_size) // pool_stride + 1 # プーリング後のサイズ(H,W共通)
        pool_output_pixel = filter_num * pool_output_size * pool_output_size # プーリング後のピクセル総数

        input_size2 = pool_output_size
        conv_output_size2 = (input_size2 + 2*filter_pad2 - filter_size2) // filter_stride2 + 1 # 畳み込み後のサイズ(H,W共通)
        pool_output_size2 = (conv_output_size2 + 2*pool_pad2 - pool_size2) // pool_stride2 + 1 # プーリング後のサイズ(H,W共通)
        pool_output_pixel2 = filter_num2 * pool_output_size2 * pool_output_size2 # プーリング後のピクセル総数

        self.use_conv2 = use_conv2
        self.use_affine2 = use_affine2
        self.use_batchnorm_C1 = use_batchnorm_C1
        self.use_batchnorm_C2 = use_batchnorm_C2
        self.use_batchnorm_A1 = use_batchnorm_A1
        self.use_batchnorm_A2 = use_batchnorm_A2
        self.use_dropout_A1 = use_dropout_A1
        self.use_dropout_A2 = use_dropout_A2
        self.dropout_ratio_A1 = dropout_ratio_A1
        self.dropout_ratio_A2 = dropout_ratio_A2
        self.use_succession = use_succession
        self.data_num = data_num
        self.prediction_mode = prediction_mode

        # if W1 == []:
        self.params = {}
        self.paramsB = {}
        std = weight_init_std

        if self.use_succession:
          #----------重みをpickleから代入--------------
          with open("params_"+str(self.data_num)+".pickle", "rb") as f:
            params_s = pickle.load(f)
          with open("params_BN"+str(self.data_num)+".pickle", "rb") as f:
            params_BN = pickle.load(f)
          # self.params = {}
          # self.paramsB = {}
          
          self.params['W1'] = params_s['W1'] # W1は畳み込みフィルターの重みになる
          self.params['b1'] = params_s['b1']
          if self.use_batchnorm_C1:
            self.paramsB["BC1_moving_mean"] = params_BN["BC1_moving_mean"]
            self.paramsB["BC1_moving_var"] = params_BN["BC1_moving_var"]
          
          if self.use_conv2:
            self.params['W1_2'] = params_s['W1_2']
            self.params['b1_2'] = params_s['b1_2']
            if self.use_batchnorm_C2:
              self.paramsB["BC2_moving_mean"] = params_BN["BC2_moving_mean"]
              self.paramsB["BC2_moving_var"] = params_BN["BC2_moving_var"]

          self.params['W2'] = params_s['W2']
          self.params['b2'] = params_s['b2']

          if self.use_batchnorm_A1:
            self.paramsB["BA1_moving_mean"] = params_BN["BA1_moving_mean"]
            self.paramsB["BA1_moving_var"] = params_BN["BA1_moving_var"]

          if self.use_affine2:
            self.params['W2_2'] = params_s['W2_2']
            self.params['b2_2'] = params_s['b2_2']
            if self.use_batchnorm_A2:
              self.paramsB["BA2_moving_mean"] = params_BN["BA2_moving_mean"]
              self.paramsB["BA2_moving_var"] = params_BN["BA2_moving_var"]

          self.params['W3'] = params_s['W3']
          self.params['b3'] = params_s['b3']

          #----------重みをpickleから代入--------------
        else:
          # 重みの初期化
          #----第１層Conv----
          self.params['W1'] = std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size) # W1は畳み込みフィルターの重みになる
          self.params['b1'] = np.zeros(filter_num) #b1は畳み込みフィルターのバイアスになる

          #----第２層Conv----
          if self.use_conv2:
            self.params['W1_2'] = std * np.random.randn(filter_num2, filter_num, filter_size2, filter_size2) #-----追加------
            self.params['b1_2'] = np.zeros(filter_num2) #-----追加------
          
          #----第３層Affine----
            self.params['W2'] = std *  np.random.randn(pool_output_pixel2, hidden_size)
          else:
            self.params['W2'] = std *  np.random.randn(pool_output_pixel, hidden_size)
          self.params['b2'] = np.zeros(hidden_size)

          #----第４層Affine----
          if self.use_affine2:
            self.params['W2_2'] = std *  np.random.randn(hidden_size, hidden_size2) #-----追加------
            self.params['b2_2'] = np.zeros(hidden_size2) #-----追加------

          #----第５層出力----
            self.params['W3'] = std *  np.random.randn(hidden_size2, output_size) #--変更-- 
          else:
            self.params['W3'] = std *  np.random.randn(hidden_size, output_size) #--変更-- 
          self.params['b3'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        #----第１層Conv----
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad']) # W1が畳み込みフィルターの重み, b1が畳み込みフィルターのバイアスになる
        if self.use_batchnorm_C1:
          print(conv_output_size)
          print(conv_output_size^2)
          batch_num = conv_output_size*conv_output_size
          if self.prediction_mode:
            self.layers['BatchNormalization_C1'] = BatchNormalization(
                np.ones(batch_num, filter_num), np.zeros(filter_num), moving_mean=self.paramsB["BC1_moving_mean"], moving_var=self.paramsB["BC1_moving_var"])
          else:
            self.layers['BatchNormalization_C1'] = BatchNormalization(np.ones(
                batch_num), np.zeros(batch_num), DataNum=self.data_num, LayerNum="C1")
            self.paramsB["BC1_moving_mean"] = self.layers['BatchNormalization_C1'].moving_mean
            self.paramsB["BC1_moving_var"] = self.layers['BatchNormalization_C1'].moving_var
        self.layers['ReLU1'] = ReLU()
        self.layers['Pool1'] = MaxPooling(pool_h=pool_size, pool_w=pool_size, stride=pool_stride, pad=pool_pad)

        #----第２層Conv----
        if self.use_conv2:
          self.layers['Conv1_2'] = Convolution(self.params['W1_2'], self.params['b1_2'],
                                            conv_param2['stride2'], conv_param2['pad2']) #-----追加------
          if self.use_batchnorm_C2:
            batch_num2 = conv_output_size2*conv_output_size2*filter_num2
            if self.prediction_mode:
              self.layers['BatchNormalization_C2'] = BatchNormalization(
                  np.ones(batch_num), np.zeros(batch_num), moving_mean=self.paramsB["BC2_moving_mean"], moving_var=self.paramsB["BC12moving_var"])
            else:
              self.layers['BatchNormalization_C2'] = BatchNormalization(np.ones(
                  batch_num), np.zeros(batch_num), DataNum=self.data_num, LayerNum="C2")
              self.paramsB["BC2_moving_mean"] = self.layers['BatchNormalization_C2'].moving_mean
              self.paramsB["BC2_moving_var"] = self.layers['BatchNormalization_C2'].moving_var
          self.layers['ReLU1_2'] = ReLU() #-----追加------
          self.layers['Pool1_2'] = MaxPooling(pool_h=pool_size2, pool_w=pool_size2, stride=pool_stride2, pad=pool_pad2) #-----追加------

        #----第３層Affine----
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        if self.use_batchnorm_A1:
          if self.prediction_mode:
            self.layers['BatchNormalization_A1'] = BatchNormalization(
                np.ones(hidden_size), np.zeros(hidden_size), moving_mean=self.paramsB["BA1_moving_mean"], moving_var=self.paramsB["BA1_moving_var"])
          else:
            self.layers['BatchNormalization_A1'] = BatchNormalization(np.ones(
                hidden_size), np.zeros(hidden_size), DataNum=self.data_num, LayerNum="A1")
            self.paramsB["BA1_moving_mean"] = self.layers['BatchNormalization_A1'].moving_mean
            self.paramsB["BA1_moving_var"] = self.layers['BatchNormalization_A1'].moving_var

        if self.use_dropout_A1:
            self.layers['DropoutA1'] = Dropout(self.dropout_ratio_A1)
        self.layers['ReLU2'] = ReLU()

        # ----第４層Affine----
        if self.use_affine2:
          self.layers['Affine2'] = Affine(self.params['W2_2'], self.params['b2_2']) #-----追加------
          if self.use_batchnorm_A2:
            if self.prediction_mode:
              self.layers['BatchNormalization_A2'] = BatchNormalization(
                  np.ones(hidden_size2), np.zeros(hidden_size2), moving_mean=self.paramsB["BA2_moving_mean"], moving_var=self.paramsB["BA2_moving_var"])
            else:
              self.layers['BatchNormalization_A2'] = BatchNormalization(np.ones(
                  hidden_size2), np.zeros(hidden_size2), DataNum=self.data_num, LayerNum="A2")
            self.paramsB["BA2_moving_mean"] = self.layers['BatchNormalization_A2'].moving_mean
            self.paramsB["BA2_moving_var"] = self.layers['BatchNormalization_A2'].moving_var

          if self.use_dropout_A2:
              self.layers['DropoutA2'] = Dropout(self.dropout_ratio_A2)
          self.layers['ReLU3'] = ReLU() #-----追加------
        
         #----第５層出力----
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.last_layer = SoftmaxWithLoss()

        # print('input size',input_size)
        # print('conv_output_size',conv_output_size)
        # print('pool_output_size',pool_output_size)
        # print('pool_output_pixel',pool_output_pixel)

        # print('input size2',input_size2)
        # print('conv_output_size2',conv_output_size2)
        # print('pool_output_size2',pool_output_size2)
        # print('pool_output_pixel2',pool_output_pixel2)

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
              x = layer.forward(x, train_flg)
            else:
              x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):

        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        if self.use_conv2:
          grads['W1_2'], grads['b1_2'] = self.layers['Conv1_2'].dW, self.layers['Conv1_2'].db #-----追加------
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        if self.use_affine2:
          grads['W2_2'], grads['b2_2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db #-----追加------
        grads['W3'], grads['b3'] = self.layers['Affine3'].dW, self.layers['Affine3'].db

        if self.prediction_mode==False:
          if self.use_batchnorm_A1:
              self.paramsB["BA1_moving_mean"] = self.layers['BatchNormalization_A1'].moving_mean
              self.paramsB["BA1_moving_var"] = self.layers['BatchNormalization_A1'].moving_var
          if self.use_batchnorm_A2:
              self.paramsB["BA2_moving_mean"] = self.layers['BatchNormalization_A2'].moving_mean
              self.paramsB["BA2_moving_var"] = self.layers['BatchNormalization_A2'].moving_var
          if self.use_batchnorm_C1:
              self.paramsB["BC1_moving_mean"] = self.layers['BatchNormalization_C1'].moving_mean
              self.paramsB["BC1_moving_var"] = self.layers['BatchNormalization_C1'].moving_var
          if self.use_batchnorm_C2:
              self.paramsB["BC2_moving_mean"] = self.layers['BatchNormalization_C2'].moving_mean
              self.paramsB["BC2_moving_var"] = self.layers['BatchNormalization_C2'].moving_var
        return grads

