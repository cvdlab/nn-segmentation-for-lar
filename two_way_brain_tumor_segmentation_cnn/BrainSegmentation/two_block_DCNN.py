import numpy as np
from keras.models import Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout, Input
from keras.layers.merge import Concatenate
from keras.optimizers import SGD
from keras import regularizers
from keras.constraints import max_norm

__author__ = "Matteo Causio"

__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Matteo Causio"
__status__ = "Production"


class TwoBlocksDCNN(object):
    """

    """

    def __init__(self, dropout_rate, learning_rate, momentum_rate, decay_rate, l1_rate, l2_rate):
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.decay_rate = decay_rate
        self.l1_rate = l1_rate
        self.l2_rate = l2_rate
        self.model = self.twoBlocksDCNN()

    # model of TwoPathCNN
    def twoBlocksDCNN(self):
        """


        :param in_channels: int, number of input channel
        :param in_shape: int, dim of the input image
        :return: Model, TwoPathCNN compiled
        """
        input = Input(shape=(65, 65, 4))
        # localPath
        locPath = Conv2D(64, (7, 7), padding='valid', activation='relu', use_bias=True,
                         kernel_regularizer=regularizers.l1_l2(self.l1_rate, self.l2_rate),
                         kernel_constraint=max_norm(2.),
                         bias_constraint=max_norm(2.))(input)
        locPath = MaxPooling2D(pool_size=(4, 4), strides=1, padding='valid')(locPath)
        locPath = Dropout(self.dropout_rate)(locPath)
        locPath = Conv2D(64, (3, 3), padding='valid', activation='relu', use_bias=True,
                         kernel_regularizer=regularizers.l1_l2(self.l1_rate, self.l2_rate),
                         kernel_constraint=max_norm(2.),
                         bias_constraint=max_norm(2.))(locPath)
        locPath = MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid')(locPath)
        locPath = Dropout(self.dropout_rate)(locPath)
        # globalPath
        globPath = Conv2D(160, (13, 13), strides=1, padding='valid', activation='relu', use_bias=True,
                          kernel_regularizer=regularizers.l1_l2(self.l1_rate, self.l2_rate),
                          kernel_constraint=max_norm(2.),
                          bias_constraint=max_norm(2.))(input)
        globPath = Dropout(self.dropout_rate)(globPath)
        # concatenation of the two path
        path = Concatenate(axis=-1)([locPath, globPath])
        # output layer
        cnn1 = Conv2D(5, (21, 21), padding='valid', activation='softmax', use_bias=True)(path)
        #second CNN
        input_cnn2 = Input(shape=(33, 33, 4))
        conc_input = Concatenate(axis=-1)([input_cnn2, cnn1])
        locPath2 = Conv2D(64, (7, 7), padding='valid', activation='relu', use_bias=True,
                         kernel_regularizer=regularizers.l1_l2(self.l1_rate, self.l2_rate),
                         kernel_constraint=max_norm(2.),
                         bias_constraint=max_norm(2.))(conc_input)
        locPath2 = MaxPooling2D(pool_size=(4, 4), strides=1, padding='valid')(locPath2)
        locPath2 = Dropout(self.dropout_rate)(locPath2)
        locPath2 = Conv2D(64, (3, 3), padding='valid', activation='relu', use_bias=True,
                         kernel_regularizer=regularizers.l1_l2(self.l1_rate, self.l2_rate),
                         kernel_constraint=max_norm(2.),
                         bias_constraint=max_norm(2.))(locPath2)
        locPath2 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid')(locPath2)
        locPath2 = Dropout(self.dropout_rate)(locPath2)
        # globalPath
        globPath2 = Conv2D(160, (13, 13), padding='valid', activation='relu', use_bias=True,
                          kernel_regularizer=regularizers.l1_l2(self.l1_rate, self.l2_rate),
                          kernel_constraint=max_norm(2.),
                          bias_constraint=max_norm(2.))(input_cnn2)
        globPath2 = Dropout(self.dropout_rate)(globPath2)
        # concatenation of the two path
        path2 = Concatenate(axis=-1)([locPath2, globPath2])
        # output layer
        output = Conv2D(5, (21, 21), strides=1, padding='valid', activation='softmax', use_bias=True)(path2)
        #compiling model
        model = Model(inputs=[input, input_cnn2], outputs=output)
        sgd = SGD(lr=self.learning_rate, momentum=self.momentum_rate, decay=self.decay_rate, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        print 'DCNN done!'
        return model
    def fit_DCNN(self, x_train, y_train, ):



if __name__ == "__main__":
    model = TwoBlocksDCNN(0.2, 0.003, 0.02, 0.00008, 0.001, 0.001)