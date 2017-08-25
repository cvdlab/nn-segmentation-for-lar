"""

The following Convolutional Neural Network it has been implemented
taking inspiration from Ruohui Wang's paper (http://www.springer.com/cda/content/
document/cda_downloaddocument/9783319406626-c2.pdf?SGWID=0-0-45-1575688-p180031493)
The patch extraction is made using the canny filter for edge detection.


"""

from __future__ import print_function
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, Dense, Flatten, Activation
from keras.initializers import glorot_normal
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.preprocessing import normalize
from skimage.exposure import adjust_sigmoid
from skimage.filters import laplace
from skimage.color import rgb2gray, gray2rgb
from skimage.exposure import adjust_gamma
from skimage.io import imread, imsave
from skimage.feature import canny as canny_filter
from skimage import img_as_float, img_as_ubyte
from glob import glob
from errno import EEXIST
from os import makedirs
from os.path import isdir
import patch_extractor_edges
import numpy as np
import argparse
import json

__author__ = "Cesare Catavitello"

__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Cesare Catavitello"
__email__ = "cesarec88@gmail.com"
__status__ = "Production"


def mkdir_p(path):
    """
    mkdir -p function, makes folder recursively if required
    :param path:
    :return:
    """
    try:
        makedirs( path )
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and isdir( path ):
            pass
        else:
            raise


# noinspection PyTypeChecker
class Edge_detector_cnn( object ):
    def __init__(self, loaded_model=False, model_name=None):
        self.loaded_model = loaded_model
        if not self.loaded_model:
            self.model = None
            self._make_model()
            self._compile_model()
        else:
            if model_name is None:
                model_to_load = str( raw_input( 'Which model should I load? ' ) )
            else:
                model_to_load = model_name
            self.model = self.load_model_weights( model_to_load )

    def _make_model(self):
        step = 0
        print( '******************************************', step )
        step += 1
        model_to_make = Sequential()
        print( '******************************************', step )
        step += 1
        model_to_make.add( Conv2D( 32, (5, 5),
                                   kernel_initializer=glorot_normal(),
                                   bias_initializer='zeros',
                                   strides=(3, 3),
                                   data_format='channels_first',
                                   input_shape=(3, 23, 23)
                                   ) )
        print( model_to_make.input_shape )
        print( model_to_make.output )
        print( '******************************************', step )
        step += 1
        model_to_make.add( Activation( 'relu' ) )
        print( model_to_make.output )
        print( '******************************************', step )
        step += 1
        model_to_make.add( Conv2D( filters=32,
                                   kernel_size=(3, 3),
                                   strides=(2, 2),
                                   data_format='channels_first',
                                   input_shape=(32, 7, 7) ) )
        print( model_to_make.output )
        print( '******************************************', step )
        step += 1
        model_to_make.add( Activation( 'relu' ) )
        print( model_to_make.output )
        print( '******************************************', 'flattened' )
        model_to_make.add( Flatten() )
        print( model_to_make.output )
        model_to_make.add( Dense( units=2, input_dim=288 ) )
        print( '******************************************', step )
        step += 1
        print( model_to_make.output )
        model_to_make.add( Activation( 'softmax' ) )
        print( '******************************************', step )
        step += 1
        print( 'model waiting to be compiled' )
        self.model = model_to_make
        print( '******************************************' )

    def fit_model(self, X_train, y_train):
        """

        :param X_train: list of patches to train on in form (n_sample, n_channel, h, w)
        :param y_train: list of labels corresponding to X_train patches in form (n_sample,)
        :return: Fits specified model
        """

        Y_train = to_categorical( y_train, 2 )

        shuffle = zip( X_train, Y_train )
        np.random.shuffle( shuffle )

        X_train = np.array( [shuffle[i][0] for i in xrange( len( shuffle ) )] )
        Y_train = np.array( [shuffle[i][1] for i in xrange( len( shuffle ) )] )

        n_epochs = 20
        self.model.fit( X_train, Y_train, epochs=n_epochs, batch_size=128, verbose=1 )

    def _compile_model(self):
        # default decay = 1e-6, lr = 0.01 maybe 1e-2 for linear decay?
        sgd = SGD( lr=3e-3,
                   decay=0,
                   momentum=0.9,
                   nesterov=True )
        print( sgd )

        self.model.compile( optimizer=sgd,
                            loss='categorical_crossentropy',
                            metrics=['accuracy'] )

    def show_segmented_image(self, index, test_img, both, canny_use=False, save=False):
        """
        Creates an image of original brain with segmentation overlay
        :param both: weather or not to use the canny filter plus segmented image,
                     and the segmentedd image only
        :param canny_use: weather or not to use the canny filter plus segmented image
        :param test_img: file path to test image for segmentation, including file extension
        :param save: If true, shows output image. (defaults to False)
        :return: if save is True, save image of segmentation results
                 if save is False, returns segmented image.
        """

        segmentation = self.predict_image( test_img )

        img_mask = np.pad( segmentation, (11, 11), mode='edge' )
        print()
        # print('+' * 40)
        # print('+', 'mask shape', img_mask.shape, '+')
        # print('+', 'mask max: {}, mask min: {}'.format(img_mask.max(), img_mask.min()), '+')
        # print('+' * 40)
        # print()

        test_back = rgb2gray( imread( test_img ).astype( 'float' ) ).reshape( 5, 216, 160 )[-2]

        edges = np.argwhere( img_mask == 1 )
        # test_back = rgb2gray(np.ones((216, 160)))
        gray_img = img_as_float( test_back )

        # adjust gamma of image
        image = adjust_gamma( gray2rgb( gray_img ), 0.8 )
        sliced_image = image.copy()

        if sliced_image.max() > 1:
            sliced_image /= sliced_image.max()

        print( '\n', 'image size : {}'.format( sliced_image.shape ) )
        color = [221. / 256, 31. / 256, 100. / 256]  # blue as choice

        print( sliced_image.shape )

        # change colors of segmented class
        for i in xrange( len( edges ) ):
            try:
                sliced_image[edges[i][0]][edges[i][1]] = color

            except:

                print( '=' * 50 )

                print( edges.max() )

                print( '=' * 50 )

                print( edges.shape )

                print( '=' * 50 )

                print( edges[i][0], edges[i][1] )
                print( sliced_image.shape )

                exit( 0 )

        canny_name = ''

        if both:
            path = './results_edge{}/'.format( canny_name )
            try:
                mkdir_p( path )
                imsave( '{}result_edge_{}{}.png'.format( path, canny_name, index ), sliced_image )
            except:
                imsave( '{}result_edge_{}{}.png'.format( path, canny_name, index ), sliced_image )
                # plt.show()
            canny_use = True

        if save:

            if canny_use:
                print( '*' * 40 )
                print( 'applying canny filter to image ' )
                canny_name = '_canny_added'
                # canny filter to the image
                canny_test_back_mask = canny_filter( test_back )
                edgess_canny = np.argwhere( canny_test_back_mask == 1 )
                for i in xrange( len( edgess_canny ) ):
                    sliced_image[edgess_canny[i][0]][edgess_canny[i][1]] = color
            path = './results_edge{}/'.format( canny_name )
            try:
                mkdir_p( path )
                imsave( '{}result_edge_{}{}.png'.format( path, canny_name, index ), sliced_image )
            except:
                imsave( '{}result_edge_{}{}.png'.format( path, canny_name, index ), sliced_image )
        else:
            return sliced_image

    def predict_image(self, test_img):
        """
        predicts classes of input image
        :param test_img: filepath to image to predict on
        :param show: displays segmentation results
        :return: segmented result
        """
        img = np.array( rgb2gray( imread( test_img ).astype( 'float' ) ).reshape( 5, 216, 160 )[-2] ) / 256

        plist = []

        # create patches from an entire slice
        img_1 = adjust_sigmoid( img ).astype( float )
        edges_1 = adjust_sigmoid( img, inv=True ).astype( float )
        edges_2 = img_1
        edges_5_n = normalize( laplace( img_1 ) )
        edges_5_n = img_as_float( img_as_ubyte( edges_5_n ) )

        plist.append( extract_patches_2d( edges_1, (23, 23) ) )
        plist.append( extract_patches_2d( edges_2, (23, 23) ) )
        plist.append( extract_patches_2d( edges_5_n, (23, 23) ) )
        patches = np.array( zip( np.array( plist[0] ), np.array( plist[1] ), np.array( plist[2] ) ) )

        # predict classes of each pixel based on model
        full_pred = self.model.predict_classes( patches )
        fp1 = full_pred.reshape( 194, 138 )
        return fp1

    def save_model(self, model_name):
        """
        Saves current model as json and weigts as h5df file
        :param model_name: name to save model and weigths under, including filepath but not extension
        :return:
        """
        model_to_save = '{}.json'.format( model_name )
        weights = '{}.hdf5'.format( model_name )
        json_string = self.model.to_json()
        try:
            self.model.save_weights( weights )
        except:
            mkdir_p( model_name )
            self.model.save_weights( weights )

        with open( model_to_save, 'w' ) as f:
            json.dump( json_string, f )

    @staticmethod
    def load_model_weights(model_name):
        """

        :param model_name: filepath to model and weights, not including extension
        :return: Model with loaded weights. can fit on model using loaded_model=True in fit_model method
        """
        print( 'Loading model {}'.format( model_name ) )
        model_to_load = '{}.json'.format( model_name )
        weights = '{}.hdf5'.format( model_name )
        with open( model_to_load ) as f:
            m = f.next()
        model_comp = model_from_json( json.loads( m ) )
        model_comp.load_weights( weights )
        print( 'Done.' )
        return model_comp


if __name__ == '__main__':

    parser = argparse.ArgumentParser( description='Commands to istanciate or load the convolutional neural network'
                                                  'for edge detection' )
    parser.add_argument( '-train',
                         '-t',
                         action='store',
                         default=1000,
                         dest='training_datas',
                         type=int,
                         help='set the number of data to train with,\n default=1000' )
    parser.add_argument( '-sigma',
                         action='store',
                         default=1,
                         dest='sigma',
                         type=float,
                         help=('set the threshold value to apply'
                               ' for the canny filter in patch extraction,\n default=1') )
    parser.add_argument( '-load',
                         '-l',
                         action='store',
                         dest='model_to_load',
                         default=0,
                         type=str,
                         help='load the model already trained,\n'
                              'default no load happen,\n'
                              'model name as:\n'
                              'model_name' )
    parser.add_argument( '-save',
                         '-s',
                         action='store_true',
                         dest='save',
                         default=False,
                         help='save the trained model in the specified path,\n'
                              'default no save happen' )
    parser.add_argument( '-augmentation',
                         '-a',
                         action='store',
                         default=0,
                         dest='angle',
                         type=int,
                         help='set data augmentation option through the specified rotating angle\n'
                              'express values in degrees, default=0' )
    parser.add_argument( '-canny',
                         '-c',
                         action='store_true',
                         dest='both',
                         default=False,
                         help=' add canny filter to segmented image (use -test option before using it)' )
    parser.add_argument( '-both',
                         '-b',
                         action='store_true',
                         dest='canny_filter',
                         default=False,
                         help=' save both canny filter to segmented image and'
                              ' segmented image (use -test option before using it(no -c is required)' )
    parser.add_argument( '-test',
                         action='store_true',
                         dest='test',
                         default=False,
                         help='execute test' )
    result = parser.parse_args()

    train_data = glob( './Training_PNG/**.png' )
    print( str( len( train_data ) ) + ' images to load' )

    if type( result.model_to_load ) is int:
        patches = patch_extractor_edges.PatchExtractor( num_samples=result.training_datas,
                                                        path_to_images=train_data,
                                                        sigma=result.sigma,
                                                        augmentation_angle=result.angle )
        X, y = patches.make_training_patches()
        model = Edge_detector_cnn()
        model.fit_model( X, y )
    else:
        model = Edge_detector_cnn( loaded_model=True, model_name='./models/' + result.model_to_load )

    if result.save:
        if result.angle is not 0:
            angle = '_augmented_{}_'.format( result.angle )
        else:
            angle = '_not_augmented_'

        model.save_model( 'models/{}_{}result_edge_detector_cnn'.format( result.training_datas, angle ) )

    if result.test:
        tests = glob( 'test_data/**' )
        segmented_images = []
        for index, slice in enumerate( tests ):
            segmented_images.append(
                model.show_segmented_image( index, slice, both=result.both, canny_use=result.canny_filter, save=True ) )
