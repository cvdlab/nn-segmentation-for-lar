import numpy as np
import random
from keras.models import Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout, Input
from keras.layers.merge import Concatenate
from keras.optimizers import SGD
from keras import regularizers
from keras.constraints import max_norm
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils

__author__ = "Matteo Causio"

__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Matteo Causio"
__status__ = "Production"


class BrainSegDCNN(object):
    """

    """

    def __init__(self, dropout_rate, learning_rate, momentum_rate, decay_rate, l1_rate, l2_rate):
        """
        The field cnn1 is initialized inside the method two_blocks_dcnn
        :param dropout_rate: rate for the dropout layer
        :param learning_rate: learning rate for training
        :param momentum_rate: rate for momentum technique
        :param decay_rate: learning rate decay over each update
        :param l1_rate: rate for l1 regularization
        :param l2_rate: rate for l2 regolarization
        """
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.decay_rate = decay_rate
        self.l1_rate = l1_rate
        self.l2_rate = l2_rate
        self.dcnn = self.two_blocks_dcnn()

    # model of TwoPathCNN
    def one_block_model(self, input_tensor):
        """
        Model for the twoPathways CNN.
        It doesn't compile the model.
        The consist of two streams, namely:
        local_path anc global_path joined
        in a final stream named path
        local_path is articulated through:
            1st convolution 64x7x7 + relu
            1st maxpooling  4x4
            1st Dropout with rate: 0.5
            2nd convolution 64x3x3 + relu
            2nd maxpooling 2x2
            2nd droput with rate: 0.5
        global_path is articulated through:
            convolution 160x13x13 + relu
            dropout with rate: 0.5
        path is articulated through:
            convolution 5x21x21

        :param input_tensor: tensor, to feed the two path
        :return: output: tensor, the output of the cnn
        """

        # localPath
        loc_path = Conv2D(64, (7, 7), padding='valid', activation='relu', use_bias=True,
                          kernel_regularizer=regularizers.l1_l2(self.l1_rate, self.l2_rate),
                          kernel_constraint=max_norm(2.),
                          bias_constraint=max_norm(2.))(input_tensor)
        loc_path = MaxPooling2D(pool_size=(4, 4), strides=1, padding='valid')(loc_path)
        loc_path = Dropout(self.dropout_rate)(loc_path)
        loc_path = Conv2D(64, (3, 3), padding='valid', activation='relu', use_bias=True,
                         kernel_regularizer=regularizers.l1_l2(self.l1_rate, self.l2_rate),
                         kernel_constraint=max_norm(2.),
                         bias_constraint=max_norm(2.))(loc_path)
        loc_path = MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid')(loc_path)
        loc_path = Dropout(self.dropout_rate)(loc_path)
        # globalPath
        glob_path = Conv2D(160, (13, 13), strides=1, padding='valid', activation='relu', use_bias=True,
                           kernel_regularizer=regularizers.l1_l2(self.l1_rate, self.l2_rate),
                           kernel_constraint=max_norm(2.),
                           bias_constraint=max_norm(2.))(input_tensor)
        glob_path = Dropout(self.dropout_rate)(glob_path)
        # concatenation of the two path
        path = Concatenate(axis=-1)([loc_path, glob_path])
        # output layer
        output = Conv2D(5, (21, 21), strides=1, padding='valid', activation='softmax', use_bias=True)(path)
        return output

    def two_blocks_dcnn(self):
        """
        Method to model and compile the first CNN and the whole two blocks DCNN.
        Also initialize the field cnn1
        :return: Model, Two blocks DeepCNN compiled
        """
        # input layers
        input65 = Input(shape=(65, 65, 4))
        input33 = Input(shape=(33, 33, 4))
        # first CNN modeling
        output_cnn1 = self.one_block_model(input65)
        # first cnn compiling
        cnn1 = Model(inputs=input65, outputs=output_cnn1)
        sgd = SGD(lr=self.learning_rate, momentum=self.momentum_rate, decay=self.decay_rate, nesterov=False)
        cnn1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        # initialize the field cnn1
        self.cnn1 = cnn1
        print 'first CNN compiled!'
        # concatenation of the output of the first CNN and the input of shape 33x33
        conc_input = Concatenate(axis=-1)([input33, output_cnn1])
        # second cnn modeling
        output_dcnn = self.one_block_model(conc_input)
        # whole dcnn compiling
        dcnn = Model(inputs=[input65, input33], outputs=output_dcnn)
        sgd = SGD(lr=self.learning_rate, momentum=self.momentum_rate, decay=self.decay_rate, nesterov=False)
        dcnn.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        print 'DCNN compiled!'
        return dcnn
    def fit_dcnn(self, x33_train, x65_train, y_train,, x33_unif_train, x65_unif_train, y_unif_train):
        Y_train = np_utils.to_categorical(y_train, 5)
        # shuffle training set
        shuffle = zip(x33_train, x65_train, Y_train)
        np.random.shuffle(shuffle)
        # transform shuffled training set back to numpy arrays
        X33_train = np.array([shuffle[i][0] for i in xrange(len(shuffle))])
        X65_train = np.array([shuffle[i][1] for i in xrange(len(shuffle))])
        Y_train = np.array([shuffle[i][2] for i in xrange(len(shuffle))]

        Y_unif_train = np_utils.to_categorical(y_unif_train, 5)
        # shuffle uniformly distribuited training set
        shuffle = zip(x33_unif_train, x65_unif_train, Y_unif_train)
        np.random.shuffle(shuffle)
        # transform shuffled uniformly distribuited training set back to numpy arrays
        X33_unif_train = np.array([shuffle[i][0] for i in xrange(len(shuffle))])
        X65_unif_train = np.array([shuffle[i][1] for i in xrange(len(shuffle))])
        Y_unif_train = np.array([shuffle[i][2] for i in xrange(len(shuffle))])

        # Stop the training if the monitor function doesn't change after patience epochs
        es = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
        # Save model after each epoch to check/bm_epoch#-val_loss
        checkpointer = ModelCheckpoint(filepath="./check/bm_{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1)
        # Fit the first cnn
        self.fit_cnn1(X33_train, Y_train, X33_unif_train, Y_unif_train )
        # Fix all the weights of the first cnn
        self.cnn1 = self.freeze_model(self.cnn1)

        fit_cnn1(X33_train, Y_train, X33_unif_train, Y_unif_train )


    def fit_cnn1(self, X33_train, Y_train, X33_unif_train, Y_unif_train):
        # Create temp cnn with input shape=(33,33,4)
        input33 = Input(shape=(33, 33, 4))
        output_cnn = self.one_block_model(input33)
        # Cnn compiling
        temp_cnn = Model(inputs=input33, outputs=output_cnn)
        sgd = SGD(lr=self.learning_rate, momentum=self.momentum_rate, decay=self.decay_rate, nesterov=False)
        temp_cnn.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
         # Stop the training if the monitor function doesn't change after patience epochs
        earlystopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
        # Save model after each epoch to check/bm_epoch#-val_loss
        checkpointer = ModelCheckpoint(filepath="./check/bm_{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1)
        # First-phase training with uniformly distribuited training set
        temp_cnn.fit(X33_train, Y_train, batch_size=self.batch_size, nb_epoch=self.nb_epoch,
                     callbacks=[earlystopping, checkpointer], validation_split=0.3, show_accuracy=True, verbose=1)
        # fix all the layers of the temporary cnn except the output layer for the second-phase
        temp_cnn = self.freeze_model(temp_cnn, freeze_output=False )
        # Second-phase training of the output layer with training set with real distribution probabily
        temp_cnn.fit(X33_unif_train, Y_unif_train, batch_size=self.batch_size, nb_epoch=self.nb_epoch,
                     callbacks=[earlystopping, checkpointer], validation_split=0.3, show_accuracy=True, verbose=1)
        # set the weights of the first cnn to the trained weights of the temporary cnn
        self.cnn1.set_weights(temp_cnn.get_weights())


    def freeze_model(self, model, freeze_output=True):
        input = model.input_layers
        output = model.output_layers
        if (freeze_model):
            n = len(model.layers)
        else:
            n = len(model.layers)-1
        for i in range(n):
            model.layers[i].trainable=False
        freezed_model=Model(inputs=input, outputs=output)
        sgd = SGD(lr=self.learning_rate, momentum=self.momentum_rate, decay=self.decay_rate, nesterov=False)
        freezed_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return freezed_model

if __name__ == "__main__":
    model = BrainSegDCNN(0.2, 0.003, 0.02, 0.00008, 0.001, 0.001)