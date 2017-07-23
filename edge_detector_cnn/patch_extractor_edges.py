"""

This code extract and catalog all patches to give to the cnn
to achieve the edge of an image through the following criteria:
0 - non edge
1 - edge

one patch is  marked as edge when passing through 2  filters(prewitt and laplacian) the  count_center function
returns a value above a certain threshold

all patches found are saved in the folder  patches/lab_{value inserted}_prew_{value inserted}/  and inside of this
two folder classes contains them together with the rotations folder each that contains all patches with the
previously specified rotation
"""

from __future__ import print_function
from random import randint
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.filters import prewitt, laplace
from skimage.color import rgb2gray
from skimage.transform import rotate
from skimage.io import imread, imsave
from errno import EEXIST
from os.path import isdir
from os import makedirs
from glob import glob
import numpy as np

__author__ = "Cesare Catavitello"

__license__ = "MIT"
__version__ = "1.0.1"
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
        makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and isdir(path):
            pass
        else:
            raise


def get_right_order(filename):
    """
    gives a key_value function for a sorted extraction
    :param filename:  path to image
    :return:
    """
    last_part = filename.split('/')[len(filename.split('/')) - 1]
    number_value = last_part[:-4]
    return int(number_value)


def count_center(edge, patch):
    """
    this function sum the value in the square of dimension 3x3
    in the center of the patch
    :param edge: the mask obtained with the filter
    :param patch: the actual patch
    :return:
    """
    sum_center = 0.0
    for k in range(-1, 1):
        for j in range(-1, 1):
            sum_center += edge[len(patch) / 2 + k][len(patch) / 2 + j]

    # return sum_center
    return sum(edge[(len(patch) / 2) - 1: (len(patch) / 2) + 1][len(patch) / 2 - 1:len(patch) / 2 + 1])


def rotate_patches(patch, edge_1, edge_2, rotating_angle):
    return np.array([rotate(patch, rotating_angle, resize=False),
                     rotate(edge_1, rotating_angle, resize=False),
                     rotate(edge_2, rotating_angle, resize=False)])


class PatchExtractor(object):
    def __init__(self, num_samples=None, path_to_images=None, lap_trsh=None, prew_trsh=None, patch_size=(23, 23),
                 augmentation_angle=0):
        """
        load and store all necessary information to achieve the patch extraction
        :param num_samples: number of patches required
        :param path_to_images: path to the folder containing all '.png.' files
        :param lap_trsh: threshold value to apply for patch extraction for what concern laplacian filter
        :param prew_trsh: threshold value to apply for patch extraction for what concern prewitt filter
        :param patch_size: dimensions for each patch
        :param augmentation_angle: angle necessary to operate the increase of the number of patches
        """
        print('*' * 50)
        print('Starting patch extraction...')
        if (lap_trsh is None) or (prew_trsh is None):
            print(" missing threshold value, impossible to proceed")
            exit(1)
        if path_to_images is None:
            ValueError('no destination file')
            exit(1)
        if num_samples is None:
            ValueError('specify the number of patches to extract and reload class')
            exit(1)
        self.laplacian_threshold = lap_trsh
        self.prewitt_threshold = prew_trsh
        self.augmentation_angle = augmentation_angle % 360
        self.images = np.array([rgb2gray(imread(path_to_images[el]).astype('float').reshape(5, 216, 160)[-2])
                                for el in range(len(path_to_images))])
        if self.augmentation_angle is not 0:
            self.augmentation_multiplier = int(360. / self.augmentation_angle)
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
        per_class = self.num_samples / len(classes)
        patches, labels = [], []
        for i in xrange(len(classes)):
            p, l = self._find_patches(classes[i], per_class)
            patches.append(p)
            labels.append(l)

        return np.array(patches).reshape((self.num_samples * self.augmentation_multiplier), 3, self.patch_size[0],
                                         self.patch_size[1]), \
               np.array(labels).reshape((self.num_samples * self.augmentation_multiplier))

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
        labels = np.ones(per_class * self.augmentation_multiplier) * class_number

        ten_percent_black = 0
        ten_percent_black_value = int(float(per_class) * 0.0001)

        start_value_extraction = 0
        full = False

        if isdir('patches/') and isdir('patches/lap_{}_prew_{}/'.format(self.laplacian_threshold,
                                                                        self.prewitt_threshold)) and isdir(
            'patches/lap_{}_prew_{}/class_{}/'.format(self.laplacian_threshold,
                                                      self.prewitt_threshold,
                                                      class_number)):

            # load all patches
            # check if quantity is enough to work
            path_to_patches = sorted(glob('./patches/lap_{}_prew_{}/class_{}/**.png'.format(self.laplacian_threshold,
                                                                                            self.prewitt_threshold,
                                                                                            class_number)),
                                     key=get_right_order)

            for path_index in xrange(len(path_to_patches)):
                if path_index < per_class:
                    patches.append(np.array(rgb2gray(imread(path_to_patches[path_index],
                                                            astype=float)).reshape(3,
                                                                                   self.patch_size[0],
                                                                                   self.patch_size[1])).astype(
                        float) / (256 * 256))
                    print('*---> patch {} loaded and added '.format(path_index))
                else:
                    full = True
                    break

            if len(path_to_patches) < per_class:
                # change start_value_extraction
                start_value_extraction = len(path_to_patches)
            else:
                full = True
        else:
            mkdir_p('patches/lap_{}_prew_{}/class_{}'.format(self.laplacian_threshold,
                                                             self.prewitt_threshold,
                                                             class_number))
        if not full:
            for i in range(start_value_extraction, per_class):
                extracted = False
                random_image = self.images[randint(0, len(self.images) - 1)]
                while np.array_equal(random_image, np.zeros(random_image.shape)):
                    random_image = self.images[randint(0, len(self.images) - 1)]
                patches_from_random = np.array(extract_patches_2d(random_image, self.patch_size, per_class))
                counter = 0

                while not extracted:
                    if counter > per_class / 2:
                        random_image = self.images[randint(0, len(self.images) - 1)]
                        patches_from_random = np.array(
                            extract_patches_2d(random_image, self.patch_size, per_class))
                        counter = 0
                    patch_1 = np.array(patches_from_random[randint(0, per_class - 1)].astype(float))
                    patch = patch_1 / (256 * 256)
                    edges_2 = prewitt(patch)
                    edges_5 = laplace(patch)
                    if edges_5.max() > 1 or edges_5.min() < -1:
                        max_value = max(edges_5.max(), -1 * edges_5.min())
                        edges_5_n = (edges_5) / max_value
                    else:
                        edges_5_n = edges_5

                    if class_number == 1:
                        first_cond = not np.array_equal(patch, np.zeros(patch.shape))
                        if first_cond:
                            second_cond = (edges_5_n[len(patch) / 2, len(patch) / 2] > self.laplacian_threshold or
                                           count_center(edges_2, patch) > self.prewitt_threshold)
                            if second_cond:
                                final_patch = np.array([patch, edges_2, edges_5_n])
                                patches.append(final_patch)
                                try:
                                    imsave('./patches/lap_{}_prew_{}/class_{}/{}.png'.format(self.laplacian_threshold,
                                                                                             self.prewitt_threshold,
                                                                                             class_number,
                                                                                             i),
                                           final_patch.reshape((3 * self.patch_size[0], self.patch_size[1])))
                                except:
                                    print(final_patch.reshape((3 * self.patch_size[0], self.patch_size[1])).max())
                                    print(final_patch.reshape((3 * self.patch_size[0], self.patch_size[1])).min())
                                    print(final_patch.reshape((3 * self.patch_size[0], self.patch_size[1])).size)
                                    exit(0)

                                print('*---> patch {} added and saved '.format(i))
                                extracted = True

                    elif class_number == 0:
                        first_cond = edges_5_n[len(patch) / 2, len(patch) / 2] <= self.laplacian_threshold and \
                                     count_center(edges_2, patch) <= self.prewitt_threshold
                        if first_cond:
                            if np.array_equal(patch, np.zeros(patch.shape)):
                                if ten_percent_black < ten_percent_black_value:
                                    final_patch = np.array([patch, edges_2, edges_5_n])
                                    patches.append(final_patch)
                                    try:
                                        imsave(
                                            './patches/lap_{}_prew_{}/class_{}/{}.png'.format(self.laplacian_threshold,
                                                                                              self.prewitt_threshold,
                                                                                              class_number,
                                                                                              i),
                                            final_patch.reshape((3 * self.patch_size[0], self.patch_size[1])))
                                    except:
                                        print(final_patch.reshape((3 * self.patch_size[0], self.patch_size[1])).max())
                                        print(final_patch.reshape((3 * self.patch_size[0], self.patch_size[1])).min())
                                        print(final_patch.reshape((3 * self.patch_size[0], self.patch_size[1])).size)
                                        exit(0)

                                    print('*---> patch {} added and saved '.format(i))
                                    ten_percent_black += 1
                                    extracted = True
                                else:
                                    pass
                            else:
                                final_patch = np.array([patch, edges_2, edges_5_n])
                                patches.append(final_patch)
                                try:
                                    imsave('./patches/lap_{}_prew_{}/class_{}/{}.png'.format(self.laplacian_threshold,
                                                                                             self.prewitt_threshold,
                                                                                             class_number,
                                                                                             i),
                                           final_patch.reshape((3 * self.patch_size[0], self.patch_size[1])))
                                except:
                                    print(final_patch.reshape((3 * self.patch_size[0], self.patch_size[1])).max())
                                    print(final_patch.reshape((3 * self.patch_size[0], self.patch_size[1])).min())
                                    print(final_patch.reshape((3 * self.patch_size[0], self.patch_size[1])).size)
                                    exit(0)

                                print('*---> patch {} added and saved '.format(i))
                                extracted = True
                    counter += 1

        if self.augmentation_angle != 0:
            print("\n *_*_*_*_* proceeding  with data augmentation for class {}  *_*_*_*_* \n".format(class_number))

            if isdir('./patches/lap_{}_prew_{}/class_{}/rotations'.format(self.laplacian_threshold,
                                                                          self.prewitt_threshold,
                                                                          class_number)):
                print("rotations folder present ")
            else:
                mkdir_p('./patches/lap_{}_prew_{}/class_{}/rotations'.format(self.laplacian_threshold,
                                                                             self.prewitt_threshold,
                                                                             class_number))
                print("rotations folder created")
            for el_index in xrange(len(patches)):
                for j in range(1, self.augmentation_multiplier):
                    try:
                        patch_rotated = np.array(rgb2gray(imread(
                            ('./patches/lap_{}_prew_{}/class_{}/'
                             'rotations/{}_{}.png'.format(self.laplacian_threshold,
                                                          self.prewitt_threshold,
                                                          class_number,
                                                          el_index,
                                                          self.augmentation_angle * j)))).reshape(3,
                                                                                                  self.patch_size[0],
                                                                                                  self.patch_size[1]
                                                                                                  )).astype(float) / (
                                            256 * 256)
                        patches.append(patch_rotated)
                        print('*---> patch {} loaded and added '
                              'with rotation of {} degrees'.format(el_index,
                                                                   self.augmentation_angle * j))
                    except:
                        final_rotated_patch = rotate_patches(patches[el_index][0],
                                                             patches[el_index][1],
                                                             patches[el_index][2],
                                                             self.augmentation_angle * j)
                        patches.append(final_rotated_patch)
                        imsave('./patches/lap_{}_prew_{}/class_{}/'
                               'rotations/{}_{}.png'.format(self.laplacian_threshold,
                                                            self.prewitt_threshold,
                                                            class_number,
                                                            el_index,
                                                            self.augmentation_angle * j),
                               final_rotated_patch.reshape(3 * self.patch_size[0], self.patch_size[1]))
                        print(('*---> patch {} saved and added '
                               'with rotation of {} degrees '.format(el_index,
                                                                     self.augmentation_angle * j)))
            print()
            print('augmentation done \n')
        print('extraction for class {} complete\n'.format(class_number))
        return np.array(patches), labels


if __name__ == '__main__':
    # path_images = glob('/Users/Cesare/Desktop/lavoro/cnn_med3d/images/Training_PNG/**')
    # prova = PatchExtractor(10, prew_trsh=0.2, lap_trsh=0.6, path_to_images=path_images)
    # patches, labels = prova.make_training_patches()
    pass
