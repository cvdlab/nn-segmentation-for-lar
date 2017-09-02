# coding=utf-8
from __future__ import print_function
from skimage.io import imsave, imread
from skimage.transform import rotate
from skimage.color import rgb2gray
from os.path import isdir
from os import makedirs
from os.path import basename
from glob import glob
from errno import EEXIST
import numpy as np
import random
import progressbar

__author__ = "Cesare Catavitello"

__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Cesare Catavitello"
__email__ = "cesarec88@gmail.com"
__status__ = "Production"

progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), progressbar.Percentage(), ' '])
np.random.seed(5)


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


def rotate_patch(patch, angle):
    """

    :param patch: patch of size (4, 33, 33)
    :param angle: says how much rotation must be applied
    :return: rotate_patch
    """

    return np.array([rotate(patch[0], angle, resize=False),
                     rotate(patch[1], angle, resize=False),
                     rotate(patch[2], angle, resize=False),
                     rotate(patch[3], angle, resize=False)])


def get_right_order(filename):
    """
    gives a key_value function for a sorted extraction
    :param filename:  path to image
    :return:
    """
    last_part = filename.split('/')[len(filename.split('/')) - 1]
    number_value = last_part[:-4]
    return int(number_value)


class PatchLibrary(object):
    """
    class for creating patches and subpatches from training data to use as input for segmentation models.
    """

    def __init__(self, patch_size=(33, 33), train_data='empty', num_samples=1000, augmentation_angle=0):
        """

        :param patch_size: tuple, size (in voxels) of patches to extract. Use (33,33) for sequential model
        :param train_data: list of filepaths to all training data saved as pngs. images should have shape (5, 216, 160)
        :param num_samples: the number of patches to collect from training data.
        :param augmentation_angle: the angle used for flipping patches(producing more datas)
        """
        if 'empty' in train_data:
            print(" insert a path for path extraction")
            exit(1)
        self.patch_size = patch_size
        if augmentation_angle % 360 != 0:
            self.augmentation_multiplier = int(float(360) / float(augmentation_angle))
        else:
            self.augmentation_multiplier = 1

        self.num_samples = num_samples
        self.augmentation_angle = augmentation_angle % 360

        self.train_data = train_data
        self.h = self.patch_size[0]
        self.w = self.patch_size[1]

    def find_patches(self, class_num, num_patches):
        """
        Helper function for sampling slices with evenly distributed classes
        :param class_num: class to sample from choice of {0, 1, 2, 3, 4}.
        :param num_patches: number of patches to extract
        :return: num_samples patches from class 'class_num' randomly selected.
        """
        h, w = self.patch_size[0], self.patch_size[1]
        patches, labels = [], np.full(num_patches * self.augmentation_multiplier, class_num, 'float')
        print('Finding patches of class {}...'.format(class_num))

        full = False
        start_value_extraction = 0
        if isdir('patches/') and isdir('patches/class_{}/'.format(class_num)):

            # load all patches
            # check if quantity is enough to work
            path_to_patches = sorted(glob('./patches/class_{}/**.png'.format(class_num)),
                                     key=get_right_order)

            for path_index in xrange(len(path_to_patches)):
                if path_index < num_patches:
                    patch_to_add = rgb2gray(imread(path_to_patches[path_index],
                                                   dtype=float)).reshape(4,
                                                                         self.patch_size[0],
                                                                         self.patch_size[1])

                    for el in xrange(len(patch_to_add)):
                        if np.max(patch_to_add[el]) > 1:
                            patch_to_add[el] = patch_to_add[el] / np.max(patch_to_add[el])

                    patches.append(patch_to_add)
                    print('*---> patch {} loaded and added '.format(path_index))
                else:
                    full = True
                    break

            if len(path_to_patches) < num_patches:
                # change start_value_extraction
                start_value_extraction = len(path_to_patches)
            else:
                full = True
        else:
            mkdir_p('patches/class_{}'.format(class_num))
        if not full:
            ct = start_value_extraction
            while ct < num_patches:
                print('searching for patch {}...'.format(ct))
                im_path = random.choice(self.train_data)
                fn = basename(im_path)
                try:
                    label = np.array(
                        imread('Labels/' + fn[:-4] + 'L.png'))
                except:
                    continue
                # resample if class_num not in selected slice
                unique, counts = np.unique(label, return_counts=True)
                labels_unique = dict(zip(unique, counts))
                try:
                    if labels_unique[class_num] < 10:
                        continue
                except:
                    continue
                # select centerpix (p) and patch (p_ix)
                img = imread(im_path).reshape(5, 216, 160)[:-1].astype('float')
                p = random.choice(np.argwhere(label == class_num))
                p_ix = (p[0] - (h / 2), p[0] + ((h + 1) / 2), p[1] - (w / 2), p[1] + ((w + 1) / 2))
                patch = np.array([i[p_ix[0]:p_ix[1], p_ix[2]:p_ix[3]] for i in img])

                # resample if patch is empty or too close to edge
                if patch.shape != (4, h, w) or len(np.argwhere(patch == 0)) > (3 * h * w):
                    if class_num == 0 and patch.shape == (4, h, w):
                        pass
                    else:
                        continue

                for slice_el in xrange(len(patch)):
                    if np.max(patch[slice_el]) != 0:
                        patch[slice_el] /= np.max(patch[slice_el])
                imsave('./patches/class_{}/{}.png'.format(class_num,
                                                          ct),
                       (np.array(patch.reshape((4 * self.patch_size[0], self.patch_size[1])))))
                patches.append(patch)
                print('*---> patch {} saved and added'.format(ct))
                ct += 1

        print()

        # for i in xrange(len(patches)):
        #     print('*' * 20)
        #     print('*' * 20)
        #
        #     print(patches[i][0].max())
        #     print(patches[i][0].min())
        #     print('*' * 20)
        #     print('*' * 20)


        if self.augmentation_angle != 0:
            print('_*_*_*_ proceed with data augmentation  for class {} _*_*_*_'.format(class_num))
            print()

            if isdir('./patches/class_{}/rotations'.format(
                    class_num)):
                print("rotations folder present ")
            else:
                mkdir_p('./patches/class_{}/rotations'.format(
                    class_num))
                print("rotations folder created")
            for el_index in xrange(len(patches)):
                for j in range(1, self.augmentation_multiplier):
                    try:
                        patch_rotated = np.array(rgb2gray(imread('./patches/class_{}/'
                                                                 'rotations/{}_{}.png'.format(class_num,
                                                                                              el_index,
                                                                                              self.augmentation_angle * j)),
                                                          dtype=float)).reshape(4,
                                                                                self.patch_size[0],
                                                                                self.patch_size[1])

                        for slice_el in xrange(len(patch_rotated)):
                            if np.max(patch_rotated[slice_el]) > 1:
                                patch_rotated[slice_el] /= np.max(patch_rotated[slice_el])

                        patches.append(patch_rotated)
                        print('*---> patch {} loaded and added '
                              'with rotation of {} degrees'.format(el_index,
                                                                   self.augmentation_angle * j))
                    except:

                        final_rotated_patch = rotate_patch(np.array(patches[el_index]), self.augmentation_angle * j)
                        patches.append(final_rotated_patch)
                        imsave('./patches/class_{}/'
                               'rotations/{}_{}.png'.format(class_num,
                                                            el_index,
                                                            self.augmentation_angle * j),
                               np.array(final_rotated_patch).reshape(4 * self.patch_size[0], self.patch_size[1]))
                        print(('*---> patch {} saved and added '
                               'with rotation of {} degrees '.format(el_index,
                                                                     self.augmentation_angle * j)))
            print()
            print('augmentation done \n')

            # for patch in patches:
            #     for i in range(1, self.augmentation_multiplier):
            #         patch_rotate = rotate_patch(patch, self.augmentation_angle * i)
            #         patches.append(patch_rotate)
            # print('data augmentation complete')
            # print()

        return np.array(patches), labels

    # def slice_to_patches(self, filename):
    #     '''
    #     Converts an image to a list of patches with a stride length of 1. Use as input for image prediction.
    #     INPUT: str 'filename': path to image to be converted to patches
    #     OUTPUT: list of patched version of imput image.
    #     '''
    #     slices = io.imread(filename).astype('float').reshape(5,240,240)[:-1]
    #     plist=[]
    #     for slice in slices:
    #         if np.max(img) != 0:
    #             img /= np.max(img)
    #         p = extract_patches_2d(img, (h,w))
    #         plist.append(p)
    #     return np.array(zip(np.array(plist[0]), np.array(plist[1]), np.array(plist[2]), np.array(plist[3])))
    #

    def make_training_patches(self, classes=None):
        """
        Creates datas(X) and labels(y) for training CNN
        :param entropy: if True, half of the patches are chosen based on highest entropy area.
        defaults to False.
        :param classes: list of classes to sample from.
         Only change default if entropy is False and balanced_classes is True
        :return:
        datas : patches (num_samples, 4_chan, h, w)
        labels (num_samples,)
        """
        if classes is None:
            classes = [0, 1, 2, 3, 4]
            per_class = self.num_samples / len(classes)
            patches, labels = [], []
            progress.currval = 0
            for i in progress(xrange(len(classes))):
                p, l = self.find_patches(classes[i], per_class)
                patches.append(p)
                labels.append(l)
            return np.array(patches).reshape(self.num_samples * self.augmentation_multiplier, 4, self.h,
                                             self.w), np.array(labels).reshape(
                self.num_samples * self.augmentation_multiplier)


if __name__ == '__main__':
    # train_data = glob('/Users/Cesare/Desktop/lavoro/cnn_med3d/images/Training_PNG/**')
    # patch_extractor = PatchLibrary(train_data=train_data, num_samples=20, augmentation_angle=180)
    # patches, labels = patch_extractor.make_training_patches()

    pass
