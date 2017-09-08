import numpy as np
import random
import json
from glob import glob
import os
import progressbar
import argparse
from patch_library import PatchLibrary
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d
from skimage import io, color, img_as_float
from skimage.exposure import adjust_gamma
from keras.models import Model, model_from_json
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout, Input, Reshape
from keras.layers.merge import Concatenate
from keras.optimizers import SGD
from keras import regularizers
from keras.constraints import max_norm
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from os import makedirs
from os.path import isdir
from errno import EEXIST


progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), progressbar.Percentage(), ' '])

def mkdir_p(path):
    """
    mkdir -p function, makes folder recursively if required
    :param path:
    :return:
    """
    try:
        makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and isdir(path):
            pass
        else:
            raise



class BrainSegDCNN(object):
    """

    """
    def __init__(self):
        self.dropout_rate = None
        self.learning_rate = None
        self.momentum_rate = None
        self.decay_rate = None
        self.l1_rate = None
        self.l2_rate = None
        self.batch_size = None
        self.nb_epoch = None
        self.nb_sample = None
        self.model = None
        self.subpatches_33 = None

    def __init__(self, dropout_rate, learning_rate, momentum_rate, decay_rate, l1_rate, l2_rate, batch_size, nb_epoch,
                 nb_sample, cascade_model=False):
        """
        The field cnn1 is initialized inside the method compile_model
        :param dropout_rate: rate for the dropout layer
        :param learning_rate: learning rate for training
        :param momentum_rate: rate for momentum technique
        :param decay_rate: learning rate decay over each update
        :param l1_rate: rate for l1 regularization
        :param l2_rate: rate for l2 regolarization
        :param batch_size:
        :param nb_epoch: number of epochs
        :param cascade_model: True if the model is input cascade
        """
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.decay_rate = decay_rate
        self.l1_rate = l1_rate
        self.l2_rate = l2_rate
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.nb_sample = nb_sample
        self.cascade_model = cascade_model
        self.model = self.compile_model()

    # model of TwoPathCNN
    def one_block_model(self, input_tensor):
        """
        Method to model one cnn. It doesn't compile the model.
        :param input_tensor: tensor, to feed the two path
        :return: output: tensor, the output of the cnn
        """

        # localPath
        loc_path = Conv2D(64, (7, 7), data_format='channels_first', padding='valid', activation='relu', use_bias=True,
                         kernel_regularizer=regularizers.l1_l2(self.l1_rate, self.l2_rate),
                         kernel_constraint=max_norm(2.),
                         bias_constraint=max_norm(2.), kernel_initializer='lecun_uniform', bias_initializer='zeros')(input_tensor)
        loc_path = MaxPooling2D(pool_size=(4, 4), data_format='channels_first', strides=1, padding='valid')(loc_path)
        loc_path = Dropout(self.dropout_rate)(loc_path)
        loc_path = Conv2D(64, (3, 3), data_format='channels_first', padding='valid', activation='relu', use_bias=True,
                          kernel_initializer='lecun_uniform', bias_initializer='zeros',
                          kernel_regularizer=regularizers.l1_l2(self.l1_rate, self.l2_rate),kernel_constraint=max_norm(2.),
                          bias_constraint=max_norm(2.))(loc_path)
        loc_path = MaxPooling2D(pool_size=(2, 2), data_format='channels_first', strides=1, padding='valid')(loc_path)
        loc_path = Dropout(self.dropout_rate)(loc_path)
        # globalPath
        glob_path = Conv2D(160, (13, 13), data_format='channels_first', strides=1, padding='valid', activation='relu', use_bias=True,
                           kernel_initializer='lecun_uniform', bias_initializer='zeros',
                           kernel_regularizer=regularizers.l1_l2(self.l1_rate, self.l2_rate),
                           kernel_constraint=max_norm(2.),
                           bias_constraint=max_norm(2.))(input_tensor)
        glob_path = Dropout(self.dropout_rate)(glob_path)
        # concatenation of the two path
        path = Concatenate(axis=1)([loc_path, glob_path])
        # output layer
        output = Conv2D(5, (21, 21), data_format='channels_first', strides=1, padding='valid', activation='softmax', use_bias=True,
                        kernel_initializer='lecun_uniform', bias_initializer='zeros')(path)
        return output

    def compile_model(self):
        """
        Model and compile the first CNN and the whole two blocks DCNN.
        Also initialize the field cnn1
        :return: Model, Two blocks DeepCNN compiled
        """
        if self.cascade_model:
            # input layers
            input65 = Input(shape=(4, 65, 65))
            input33 = Input(shape=(4, 33, 33))
            # first CNN modeling
            output_cnn1 = self.one_block_model(input65)
            # first cnn compiling
            cnn1 = Model(inputs=input65, outputs=output_cnn1)
            sgd = SGD(lr=self.learning_rate, momentum=self.momentum_rate, decay=self.decay_rate, nesterov=False)
            cnn1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            # initialize the field cnn1
            self.cnn1 = cnn1
            print 'First CNN compiled!'
            # concatenation of the output of the first CNN and the input of shape 33x33
            conc_input = Concatenate(axis=1)([input33, output_cnn1])
            # second cnn modeling
            output_dcnn = self.one_block_model(conc_input)
            output_dcnn = Reshape((5,))(output_dcnn)
            # whole dcnn compiling
            dcnn = Model(inputs=[input65, input33], outputs=output_dcnn)
            sgd = SGD(lr=self.learning_rate, momentum=self.momentum_rate, decay=self.decay_rate, nesterov=False)
            dcnn.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            print 'Cascade DCNN compiled!'
            return dcnn
        else:
            # input layers
            input33 = Input(shape=(4, 33, 33))
            # first CNN modeling
            output_cnn1 = self.one_block_model(input33)
            # first cnn compiling
            cnn1 = Model(inputs=input33, outputs=output_cnn1)
            sgd = SGD(lr=self.learning_rate, momentum=self.momentum_rate, decay=self.decay_rate, nesterov=False)
            cnn1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            # initialize the field cnn1
            self.cnn1 = cnn1
            print 'Two pathway CNN compiled!'
            return cnn1

    def fit_model(self, x33_train, y_train, x33_uniftrain, y_uniftrain, x65_train=None, x65_uniftrain=None):
        '''
        Fit the model in both modality single or cascade. For cascade model need either 65x65 and 33x33 patches
        :param x33_train:33x33 patches
        :param x65_train:65x65 patches
        :param y_train: labels
        :param x33_uniftrain:33x33 uniformly distribuited patches
        :param x65_uniftrain:65x65 uniformly distribuited patches
        :param y_uniftrain:uniformly distribuited labels
        '''
        if self.cascade_model:
            if x65_train == None and x65_uniftrain == None:
                print 'Error: patches 65x65, necessary to fit cascade model, not inserted.'
            X33_train, X65_train, Y_train, X33_uniftrain, X65_uniftrain, Y_uniftrain = self.init_cascade_training(x33_train,
                                    x65_train, y_train, x33_uniftrain, x65_uniftrain, y_uniftrain)
            # Stop the training if the monitor function doesn't change after patience epochs
            earlystopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
            # Save model after each epoch to check/bm_epoch#-val_loss
            checkpointer = ModelCheckpoint(filepath="/home/ixb3/Scrivania/check/bm_{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1)
            # Fit the first cnn
            self.fit_cnn1(X33_train, Y_train, X33_uniftrain, Y_uniftrain)
            # Fix all the weights of the first cnn
            self.cnn1 = self.freeze_model(self.cnn1)

            # First-phase training of the second cnn
            self.model.fit(x=[X65_uniftrain, X33_uniftrain], y=Y_uniftrain, batch_size=self.batch_size, epochs=self.nb_epoch,
                       callbacks=[earlystopping, checkpointer], validation_split=0.3, verbose=1)
            # fix all the layers of the dcnn except the output layer for the second-phase
            self.freeze_model(self.model, freeze_output=False)
            # Second-phase training of the second cnn
            self.model.fit(x=[X65_train, X33_train], y=Y_uniftrain, batch_size=self.batch_size,
                       epochs=self.nb_epoch,
                       callbacks=[earlystopping, checkpointer], validation_split=0.3, verbose=1)
            print 'Model trained'
        else:
            X33_train, Y_train, X33_uniftrain, Y_uniftrain = self.init_single_training(x33_train, y_train,
                                                                                x33_uniftrain, y_uniftrain)
            self.fit_cnn1(X33_train, Y_train, X33_uniftrain, Y_uniftrain)
            self.model = self.cnn1
            print 'Model trained'

    def fit_cnn1(self, X33_train, Y_train, X33_unif_train, Y_unif_train):
        # Create temp cnn with input shape=(4,33,33,)
        input33 = Input(shape=(4, 33, 33))
        output_cnn = self.one_block_model(input33)
        output_cnn = Reshape((5,))(output_cnn)
        # Cnn compiling
        temp_cnn = Model(inputs=input33, outputs=output_cnn)
        sgd = SGD(lr=self.learning_rate, momentum=self.momentum_rate, decay=self.decay_rate, nesterov=False)
        temp_cnn.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        # Stop the training if the monitor function doesn't change after patience epochs
        earlystopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
        # Save model after each epoch to check/bm_epoch#-val_loss
        checkpointer = ModelCheckpoint(filepath="/home/ixb3/Scrivania/check/bm_{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1)
        # First-phase training with uniformly distribuited training set
        temp_cnn.fit(x=X33_train, y=Y_train, batch_size=self.batch_size, epochs=self.nb_epoch,
                     callbacks=[earlystopping, checkpointer], validation_split=0.3,  verbose=1)
        # fix all the layers of the temporary cnn except the output layer for the second-phase
        temp_cnn = self.freeze_model(temp_cnn, freeze_output=False)
        # Second-phase training of the output layer with training set with real distribution probabily
        temp_cnn.fit(x=X33_unif_train, y=Y_unif_train, batch_size=self.batch_size, epochs=self.nb_epoch,
                     callbacks=[earlystopping, checkpointer], validation_split=0.3, verbose=1)
        # set the weights of the first cnn to the trained weights of the temporary cnn
        self.cnn1.set_weights(temp_cnn.get_weights())

    def freeze_model(self, compiled_model, freeze_output=True):
        '''
        Freeze the weights of the model, they will not be adjusted during training
        :param compiled_model: model to freeze
        :param freeze_output: if false the weights of the last layer of the model will not be freezed
        :return: model with freezed weights
        '''
        input_layer = compiled_model.inputs
        output_layer = compiled_model.outputs
        if freeze_output:
            n = len(compiled_model.layers)
        else:
            n = len(compiled_model.layers) - 1
        for i in range(n):
            compiled_model.layers[i].trainable = False
        freezed_model = Model(inputs=input_layer, outputs=output_layer)
        sgd = SGD(lr=self.learning_rate, momentum=self.momentum_rate, decay=self.decay_rate, nesterov=False)
        freezed_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        print 'Model freezed'
        return freezed_model

    def init_single_training(self, x3, y, x3_unif, y_unif):
        '''
        helper function to initialize the training of the single model: shuffle the training set and make categorical
        the targets
        :param x3: 33x33 patches
        :param y: labels
        :param x3_unif: 33x33 uniformly distribuited patches
        :param y_unif: uniformly distribuited labels
        :return:
        '''
        Y_train = np_utils.to_categorical(y, 5)
        # shuffle training set
        shuffle = zip(x3, Y_train)
        np.random.shuffle(shuffle)
        # transform shuffled training set back to numpy arrays
        X33_train = np.array([shuffle[i][0] for i in xrange(len(shuffle))])
        Y_train = np.array([shuffle[i][1] for i in xrange(len(shuffle))])  # .reshape((len(shuffle), 5, 1, 1))
        Y_uniftrain = np_utils.to_categorical(y_unif, 5)
        # shuffle uniformly distributed training set
        shuffle = zip(x3_unif, Y_uniftrain)
        np.random.shuffle(shuffle)
        # transform shuffled uniformly distribuited training set back to numpy arrays
        X33_uniftrain = np.array([shuffle[i][0] for i in xrange(len(shuffle))])
        Y_uniftrain = np.array([shuffle[i][1] for i in xrange(len(shuffle))])  # .reshape((len(shuffle), 5, 1, 1))
        return X33_train, Y_train, X33_uniftrain, Y_uniftrain

    def init_cascade_training(self, x3, x6, y, x3_unif, x6_unif, y_unif):
        '''
        helper function to initialize the training of the cascade model:: shuffle the training set and make categorical
        the targets
        :param x3: 33x33 patches
        :param x6: 65x65 patches
        :param y: labels
        :param x3_unif: 33x33 uniformly distribuited patches
        :param x6_unif: 65x65 uniformly distribuited patches
        :param y_unif: uniformly distribuited labels
        :return:
        '''
        Y_train = np_utils.to_categorical(y, 5)
        # shuffle training set
        shuffle = zip(x3, x6, Y_train)
        np.random.shuffle(shuffle)
        # transform shuffled training set back to numpy arrays
        X33_train = np.array([shuffle[i][0] for i in xrange(len(shuffle))])
        X65_train = np.array([shuffle[i][1] for i in xrange(len(shuffle))])
        Y_train = np.array([shuffle[i][2] for i in xrange(len(shuffle))])  # .reshape((len(shuffle), 5, 1, 1))
        Y_uniftrain = np_utils.to_categorical(y_unif, 5)
        # shuffle uniformly distribuited training set
        shuffle = zip(x3_unif, x6_unif, Y_uniftrain)
        np.random.shuffle(shuffle)
        # transform shuffled uniformly distribuited training set back to numpy arrays
        X33_uniftrain = np.array([shuffle[i][0] for i in xrange(len(shuffle))])
        X65_uniftrain = np.array([shuffle[i][1] for i in xrange(len(shuffle))])
        Y_uniftrain = np.array([shuffle[i][2] for i in xrange(len(shuffle))])  # .reshape((len(shuffle), 5, 1, 1))
        return X33_train, X65_train, Y_train, X33_uniftrain, X65_uniftrain, Y_uniftrain

    def save_model(self, model_name):
        '''
        INPUT string 'model_name': path where to save model and weights, without extension
        Saves current model as json and weights as h5df file
        '''

        model_tosave = '{}.json'.format(model_name)
        weights = '{}.hdf5'.format(model_name)
        json_string = self.model.to_json()
        self.model.save_weights(weights)
        with open(model_tosave, 'w') as f:
            json.dump(json_string, f)
        print 'Model saved.'

    def load_model(self, model_name):
        '''
        Load a model
        INPUT  (1) string 'model_name': filepath to model and weights, not including extension
        OUTPUT: Model with loaded weights. can fit on model using loaded_model=True in fit_model method
        '''
        print 'Loading model {}'.format(model_name)
        model_toload = '{}.json'.format(model_name)
        weights = '{}.hdf5'.format(model_name)
        with open(model_toload) as f:
            m = f.next()
        model_comp = model_from_json(json.loads(m))
        model_comp.load_weights(weights)
        print 'Model loaded.'
        self.model = model_comp
        return model_comp

    def predict_image(self, filepath_image, show=False):
        '''
        predicts classes of input image
        INPUT   (1) str 'filepath_image': filepath to image to predict on
                (2) bool 'show': True to show the results of prediction, False to return prediction
        OUTPUT  (1) if show == False: array of predicted pixel classes for the center 208 x 208 pixels
                (2) if show == True: displays segmentation results
        '''
        print 'Starting prediction...'
        if self.cascade_model:
            images = io.imread(filepath_image).astype('float').reshape(5, 216, 160)
            p33list = []
            p65list = []
            # create patches from an entire slice
            for image in images[:-1]:
                if np.max(image) != 0:
                    image /= np.max(image)
                patch65 = extract_patches_2d(image, (65, 65))
                p65list.append(patch65)
                p33list.append(self.center_n(33, patch65))
                print str(len(p33list))
            patches33 = np.array(zip(p33list[0], p33list[1], p33list[2], p33list[3]))
            patches65 = np.array(zip(p65list[0], p65list[1], p65list[2], p65list[3]))
            # predict classes of each pixel based on model
            prediction = self.model.predict([patches65, patches33])
            print 'Predicted'
            prediction = prediction.reshape(208, 208)
            if show:
                io.imshow(prediction)
                plt.show
            else:
                return prediction
        else:
            images = io.imread(filepath_image).astype('float').reshape(5, 216, 160)
            p33list = []
            # create patches from an entire slice
            for image in images[:-1]:
                if np.max(image) != 0:
                    image /= np.max(image)
                patch33 = extract_patches_2d(image, (33, 33))
                p33list.append(patch33)
            patches33 = np.array(zip(p33list[0], p33list[1], p33list[2], p33list[3]))
            # predict classes of each pixel based on model
            prediction = self.cnn1.predict(patches33)
            print 'Predicted'
            prediction = prediction.reshape(5, 184, 128)
            predicted_classes = np.argmax(prediction, axis=0)
            if show:
                print 'Let s show'
                for i in range(5):
                    io.imshow(prediction[i])
                    plt.show
                    print 'Showed'
                    return prediction
            else:
                return predicted_classes



    def center_n(self, n, patches):
        """
        helper function, get the nxn centered subpatch
        :param n: int, size of center patch to take (square)
        :param patches: list of patches to take subpatch of
        :return: list of center nxn patches.
        """
        h = 65
        w = 65
        sub_patches = []
        for mode in patches:
            subs = np.array(

                    mode[(h / 2) - (n / 2):(h / 2) + ((n + 1) / 2),
                    (w / 2) - (n / 2):(w / 2) + ((n + 1) / 2)]
                    #for patch in mode

            )
            sub_patches.append(subs)
        return np.array(sub_patches)

    def save_segmented_image(self, filepath_image, modality='t1c', show=False):
        '''
        Creates an image of original brain with segmentation overlay and save it in ./predictions
        INPUT   (1) str 'filepath_image': filepath to test image for segmentation, including file extension
                (2) str 'modality': imaging modality to use as background. defaults to t1c. options: (flair, t1, t1c, t2)
                (3) bool 'show': If true, shows output image. defaults to False.
        OUTPUT  (1) if show is True, shows image of segmentation results
                (2) if show is false, returns segmented image.
        '''
        modes = {'flair': 0, 't1': 1, 't1c': 2, 't2': 3}

        segmentation = self.predict_image(filepath_image, show=False)
        print 'segmentation = ' + str(segmentation)
        img_mask = np.pad(segmentation, (16, 16), mode='edge')
        ones = np.argwhere(img_mask == 1)
        twos = np.argwhere(img_mask == 2)
        threes = np.argwhere(img_mask == 3)
        fours = np.argwhere(img_mask == 4)

        test_im = io.imread(filepath_image)
        test_back = test_im.reshape(5, 216, 160)[modes[modality]]
        # overlay = mark_boundaries(test_back, img_mask)
        gray_img = img_as_float(test_back)

        # adjust gamma of image
        image = adjust_gamma(color.gray2rgb(gray_img), 0.65)
        sliced_image = image.copy()
        red_multiplier = [1, 0.2, 0.2]
        yellow_multiplier = [1, 1, 0.25]
        green_multiplier = [0.35, 0.75, 0.25]
        blue_multiplier = [0, 0.25, 0.9]

        print str(len(ones))
        print str(len(twos))
        print str(len(threes))
        print str(len(fours))

        # change colors of segmented classes
        for i in xrange(len(ones)):
            sliced_image[ones[i][0]][ones[i][1]] = red_multiplier
        for i in xrange(len(twos)):
            sliced_image[twos[i][0]][twos[i][1]] = green_multiplier
        for i in xrange(len(threes)):
            sliced_image[threes[i][0]][threes[i][1]] = blue_multiplier
        for i in xrange(len(fours)):
            sliced_image[fours[i][0]][fours[i][1]] = yellow_multiplier
        #if show=True show the prediction
        if show:
            print 'Showing...'
            io.imshow(sliced_image)
            plt.show()
        #save the prediction
        print 'Saving...'
        try:
            mkdir_p('./predictions/')
            io.imsave('./predictions/' + os.path.basename(filepath_image) + '.png', sliced_image)
            print 'prediction saved.'
        except:
            io.imsave('./predictions/' + os.path.basename(filepath_image) + '.png', sliced_image)
            print 'prediction saved.'

if __name__ == "__main__":
    #set arguments
    parser = argparse.ArgumentParser(description='Commands to istanciate or load the convolutional neural network')
    parser.add_argument('-cascade',
                        '-c',
                        action='store',
                        default=False,
                        dest='cascade_model',
                        type=bool,
                        help='set the model to be cascade_model(True) or single_model(False)\n default=False')

    parser.add_argument('-train',
                        '-t',
                        action='store',
                        default=1000,
                        dest='nb_samples',
                        type=int,
                        help='set the number of data to train with,\n default=1000')
    parser.add_argument('-samplespath',
                        '-sp',
                        action='store',
                        default='Training_PNG',
                        dest='training_path',
                        type=str,
                        help='set the path of the folder containing samples for training,\n default=None')
    parser.add_argument('-labelspath',
                        '-lp',
                        action='store',
                        default='Labels',
                        dest='labels_path',
                        type=str,
                        help='set the path of the folder containing labels for training,\n default=None')

    parser.add_argument('-load',
                        '-l',
                        action='store',
                        dest='model_to_load',
                        default=0,
                        type=str,
                        help='load the model already trained,\n'
                             'default no load happen,\n'
                             'model name as:\n'
                             'model_name')
    parser.add_argument('-save',
                        '-s',
                        action='store',
                        dest='save',
                        default=None,
                        type=str,
                        help='save the model after been trained,\n'
                             'default no save happen,\n'
                             'model name as:\n'
                             'model_name')
    parser.add_argument('-test',
                        action='store',
                        dest='test_path',
                        default=None,
                        type=str,
                        help='path of the folder containing the RMI images to execute the brain segmentation\n'
                             'default no execution happens,\n'
                             'test folder path as:\n'
                             'test_path')
    result = parser.parse_args()
    #compile the model
    brain_seg = BrainSegDCNN(dropout_rate=0.2, learning_rate=0.01, momentum_rate=0.5, decay_rate=0.1,
                                 l1_rate=0.001,
                                 l2_rate=0.001, batch_size=20, nb_epoch=10, nb_sample=result.nb_samples,
                                 cascade_model=result.cascade_model)
    #train the model if -load is not given
    if type(result.model_to_load) is int:
        # check if the path of the folder containing the training samples is given, otherwise ask to insert it
        training_set = glob(result.training_path + '/**')
        #train the model in single or cascade model
        if brain_seg.cascade_model:
            patches = PatchLibrary(train_samples=training_set, label_folder_path=result.labels_path,
                                       num_samples=brain_seg.nb_sample)
            x33_train, x65_train, y_train = patches.make_training_patches(balanced_classes=False)
            x33_uniftrain, x65_uniftrain, y_uniftrain = patches.make_training_patches()
            # fit model
            brain_seg.fit_model(x33_train, y_train, x33_uniftrain, y_uniftrain, x65_train=x65_train,
                                    x65_uniftrain=x65_uniftrain)
        else:
            patches = PatchLibrary(train_samples=training_set, label_folder_path=label_folder_path,
                                   num_samples=brain_seg.nb_sample, patch_size=(33, 33), subpatches_33=False)
            x33_train, y_train = patches.make_training_patches(balanced_classes=False)
            x33_uniftrain, y_uniftrain = patches.make_training_patches()
            # fit model
            brain_seg.fit_model(x33_train, y_train, x33_uniftrain, y_uniftrain)
    #load an already trained model if -load is given
    else:
        brain_seg.model = brain_seg.load_model('./models/' + result.model_to_load)
    #save the model if -save is given
    if result.save is not None:
        brain_seg.save_model('./models/' + result.save)
    #predict all the RMI images contained in the given path -test
    if result.test is not None:
        tests = glob(result.test + '/**')
        segmented_images = []
        for topredict_img in tests:
            brain_seg.save_segmented_image(topredict_img, show=True)
