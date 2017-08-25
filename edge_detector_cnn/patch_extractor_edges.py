"""
This code extract and catalog all patches to give to the cnn
to achieve the edge of an image through the following criteria:
0 - non edge
1 - edge
one patch is  marked as edge when passing through the canny edge extractor
all patches found are saved in the folder  patches/sigma_{value inserted}/  and inside of this
two folder classes contains them together with the rotations folder each that contains all patches with the
previously specified rotation
"""

from __future__ import print_function
from random import randint
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.preprocessing import normalize
from skimage.color import rgb2gray
from skimage.exposure import adjust_sigmoid
from skimage.transform import rotate
from skimage.filters import laplace
from skimage.feature import canny
from skimage.io import imread, imsave
from skimage import img_as_ubyte, img_as_float
from errno import EEXIST
from os.path import isdir
from os import makedirs
from glob import glob
import numpy as np

__author__ = "Cesare Catavitello"

__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Cesare Catavitello"
__email__ = "cesarec88@gmail.com"
__status__ = "Production"


def mkdir_p(path):
    """
    mkdir -p function, makes folder recursively if required
    :type path: basestring
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


def get_right_order(filename):
    """
    gives a key_value function for a sorted extraction
    :param filename:  path to image
    :return:
    """
    last_part = filename.split( '/' )[len( filename.split( '/' ) ) - 1]
    number_value = last_part[:-4]
    return int( number_value )

def is_boarder(patch, sigma):
    """
    this function evaluates the canny filter ovet the passed patch
    returning through boolean
    wheather or not the image can be catalogued as boarder retrieving the
    boolean value of the center
    :param patch: the image to filter
    :param sigma: sigma value for the canny filter
    :return: boolean value
    """
    bool_patch = canny( patch, sigma=sigma )
    return bool_patch[patch.shape[0] / 2][patch.shape[1] / 2]


def rotate_patches(patch, edge_1, edge_2, rotating_angle):
    return np.array( [rotate( patch, rotating_angle, resize=False ),
                      rotate( edge_1, rotating_angle, resize=False ),
                      rotate( edge_2, rotating_angle, resize=False )] )


class PatchExtractor( object ):
    def __init__(self, num_samples=None, path_to_images=None, sigma=None,
                 patch_size=(23, 23),
                 augmentation_angle=0):
        """
        load and store all necessary information to achieve the patch extraction
        :param num_samples: number of patches required
        :param path_to_images: path to the folder containing all '.png.' files
        :param sigma: sigma value to apply at the canny filter for patch extraction criteria
        :param patch_size: dimensions for each patch
        :param augmentation_angle: angle necessary to operate the increase of the number of patches
        """
        print( '*' * 50 )
        print( 'Starting patch extraction...' )
        # inserire il check per sigma
        if sigma is None:
            print( " missing threshold value, impossible to proceed" )
            exit( 1 )
        if path_to_images is None:
            ValueError( 'no destination file' )
            exit( 1 )
        if num_samples is None:
            ValueError( 'specify the number of patches to extract and reload class' )
            exit( 1 )
        self.sigma = sigma
        self.augmentation_angle = augmentation_angle % 360
        self.images = np.array( [rgb2gray( imread( path_to_images[el] ).astype( 'float' ).reshape( 5, 216, 160 )[-2] )
                                 for el in range( len( path_to_images ) )] )
        if self.augmentation_angle is not 0:
            self.augmentation_multiplier = int( 360. / self.augmentation_angle )
        else:
            self.augmentation_multiplier = 1

        self.num_samples = num_samples

        self.patch_size = patch_size

    def make_training_patches(self):
        """
        Creates datas(all patches) and labels for training CNN
        :return:
        datas : patches (num_samples, 3_chan, height, wide)
        labels (num_samples,)
        """
        classes = [0, 1]
        per_class = self.num_samples / len( classes )
        patches, labels = [], []

        for i in xrange( len( classes ) ):
            p, l = self._find_patches( classes[i], per_class )
            patches.append( p )
            labels.append( l )

        return np.array( patches ).reshape( (self.num_samples * self.augmentation_multiplier), 3, self.patch_size[0],
                                            self.patch_size[1] ), \
               np.array( labels ).reshape( (self.num_samples * self.augmentation_multiplier) )

    def _find_patches(self, class_number, per_class):
        """
        this function in dependence of the class number search for patches with the edge pattern
        :param per_class: number of patches to find
        :param class_number the class number
        :return:
        a numpy array of patches and a numpy array of labels
        """
        print()
        patches = []
        labels = np.ones( per_class * self.augmentation_multiplier ) * class_number

        ten_percent_black = 0
        ten_percent_black_value = int( float( per_class ) * 0.0001 )

        start_value_extraction = 0
        full = False

        if isdir( 'patches/' ) and isdir( 'patches/sigma_{}/'.format( self.sigma
                                                                      ) ) and isdir(
            'patches/sigma_{}/class_{}/'.format( self.sigma,
                                                 class_number ) ):

            # load all patches
            # check if quantity is enough to work
            path_to_patches = sorted( glob( './patches/sigma_{}/class_{}/**.png'.format( self.sigma,
                                                                                         class_number ) ),
                                      key=get_right_order )

            for path_index in xrange( len( path_to_patches ) ):
                if path_index < per_class:
                    patch_to_load = np.array( rgb2gray( imread( path_to_patches[path_index],
                                                                dtype=float ) ).reshape( 3,
                                                                                         self.patch_size[0],
                                                                                         self.patch_size[1] ) ).astype(
                        float )
                    patches.append( patch_to_load )
                    for el in xrange( len( patch_to_load ) ):
                        if np.max( patch_to_load[el] ) > 1:
                            patch_to_load[el] /= np.max( patch_to_load[el] )
                    print( '*---> patch {}/{} loaded and added '.format( path_index, per_class ) )
                else:
                    full = True
                    break

            if len( path_to_patches ) < per_class:
                # change start_value_extraction
                start_value_extraction = len( path_to_patches )
            else:
                full = True
        else:
            mkdir_p( 'patches/sigma_{}/class_{}'.format( self.sigma,
                                                         class_number ) )

        patch_to_extract = 25000

        if not full:
            for i in range( start_value_extraction, per_class ):
                extracted = False
                random_image = self.images[randint( 0, len( self.images ) - 1 )]
                while np.array_equal( random_image, np.zeros( random_image.shape ) ):
                    random_image = self.images[randint( 0, len( self.images ) - 1 )]
                patches_from_random = np.array( extract_patches_2d( random_image, self.patch_size, patch_to_extract ) )
                counter = 0

                while not extracted:
                    if counter > per_class / 2:
                        random_image = self.images[randint( 0, len( self.images ) - 1 )]
                        patches_from_random = np.array(
                            extract_patches_2d( random_image, self.patch_size, patch_to_extract ) )
                        counter = 0
                    patch = np.array( patches_from_random[randint( 0, patch_to_extract - 1 )].astype( float ) )
                    if patch.max() > 1:
                        patch = normalize( patch )

                    patch_1 = adjust_sigmoid( patch )
                    edges_1 = adjust_sigmoid( patch, inv=True )
                    edges_2 = patch_1
                    edges_5_n = normalize( laplace( patch ) )
                    edges_5_n = img_as_float( img_as_ubyte( edges_5_n ) )

                    choosing_cond = is_boarder( patch=patch_1, sigma=self.sigma )

                    if class_number == 1 and choosing_cond:
                        final_patch = np.array( [edges_1, edges_2, edges_5_n] )
                        patches.append( final_patch )
                        try:
                            imsave( './patches/sigma_{}/class_{}/{}.png'.format( self.sigma,
                                                                                 class_number,
                                                                                 i ),
                                    final_patch.reshape( (3 * self.patch_size[0], self.patch_size[1]) ),
                                    dtype=float )
                        except:
                            print( 'problem occurred in save for class {}'.format( class_number ) )
                            exit( 0 )

                        print( '*---> patch {}/{} added and saved '.format( i, per_class ) )
                        extracted = True

                    elif class_number == 0 and not choosing_cond:
                        if np.array_equal( patch, np.zeros( patch.shape ) ):
                            if ten_percent_black < ten_percent_black_value:
                                final_patch = np.array( [patch, edges_2, edges_5_n] )
                                patches.append( final_patch )
                                try:

                                    imsave( './patches/sigma_{}/class_{}/{}.png'.format( self.sigma,
                                                                                         class_number,
                                                                                         i ),
                                            final_patch.reshape( (3 * self.patch_size[0], self.patch_size[1]) ),
                                            dtype=float )
                                except:
                                    print( 'problem occurred in save for class {}'.format( class_number ) )
                                    exit( 0 )

                                print( '*---> patch {}/{} added and saved '.format( i, per_class ) )
                                ten_percent_black += 1
                                extracted = True
                            else:
                                pass
                        else:
                            final_patch = np.array( [edges_1, edges_2, edges_5_n] )
                            patches.append( final_patch )
                            try:

                                imsave( './patches/sigma_{}/class_{}/{}.png'.format( self.sigma,
                                                                                     class_number,
                                                                                     i ),
                                        final_patch.reshape( (3 * self.patch_size[0], self.patch_size[1]) ),
                                        dtype=float )
                            except:
                                print( 'problem occurred in save for class {}'.format( class_number ) )
                                exit( 0 )

                            print( '*---> patch {}/{} added and saved '.format( i, per_class ) )
                            extracted = True
                    counter += 1

        if self.augmentation_angle != 0:
            print( "\n *_*_*_*_* proceeding  with data augmentation for class {}  *_*_*_*_* \n".format( class_number ) )

            if isdir( './patches/sigma_{}/class_{}/rotations'.format( self.sigma,
                                                                      class_number ) ):
                print( "rotations folder present " )
            else:
                mkdir_p( './patches/sigma_{}/class_{}/rotations'.format( self.sigma,
                                                                         class_number ) )
                print( "rotations folder created" )
            for el_index in xrange( len( patches ) ):
                for j in range( 1, self.augmentation_multiplier ):
                    try:
                        patch_rotated = np.array( rgb2gray( imread(
                            ('./patches/sigma_{}/class_{}/'
                             'rotations/{}_{}.png'.format( self.sigma,
                                                           class_number,
                                                           el_index,
                                                           self.augmentation_angle * j )) ) ).reshape( 3,
                                                                                                       self.patch_size[
                                                                                                           0],
                                                                                                       self.patch_size[
                                                                                                           1]
                                                                                                       ) ).astype(
                            float ) / (
                                            256 * 256)
                        patches.append( patch_rotated )
                        print( '*---> patch {}/{} loaded and added '
                               'with rotation of {} degrees'.format( el_index, per_class,
                                                                     self.augmentation_angle * j ) )
                    except:
                        final_rotated_patch = rotate_patches( patches[el_index][0],
                                                              patches[el_index][1],
                                                              patches[el_index][2],
                                                              self.augmentation_angle * j )
                        patches.append( final_rotated_patch )
                        imsave( './patches/sigma_{}/class_{}/'
                                'rotations/{}_{}.png'.format( self.sigma,
                                                              class_number,
                                                              el_index,
                                                              self.augmentation_angle * j ),
                                final_rotated_patch.reshape( 3 * self.patch_size[0], self.patch_size[1] ),
                                dtype=float )
                        print( ('*---> patch {}/{} saved and added '
                                'with rotation of {} degrees '.format( el_index, per_class,
                                                                       self.augmentation_angle * j )) )
            print()
            print( 'augmentation done \n' )
        print( 'extraction for class {} complete\n'.format( class_number ) )
        return np.array( patches ), labels


if __name__ == '__main__':
    # path_images = glob( '/Users/Cesare/Desktop/lavoro/cnn_med3d/images/Training_PNG/**' )
    # prova = PatchExtractor( 40, sigma=.6, path_to_images=path_images )
    # patches, labels = prova.make_training_patches()
    pass
