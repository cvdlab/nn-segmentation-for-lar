"""


This code pick randomly an image between all .mha images (picked up randomly) in the specified folder and
convert it into .png image  in accordance to the number of images required.


"""

from __future__ import print_function
import SimpleITK.SimpleITK as sitk
import matplotlib.pyplot as plt
import random as rnd
import numpy as np
from glob import glob
from os import makedirs
from os.path import isdir
from errno import EEXIST

__author__ = "Cesare Catavitello"
__license__ = "MIT"
__version__ = "1.0.1"
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
        makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and isdir(path):
            pass
        else:
            raise


class ImagePngConverter:
    """
    a class to convert an .mha slice into .png image
    to compute a random test with an input image to search for tumor patterns
    """

    def __init__(self, global_counter, path_to_mha=None, how_many_from_one=1, saving_path='./test_data/'):
        if path_to_mha is None:
            raise NameError(' missing .mha path ')
        self.images = []
        for i in range(0, len(path_to_mha)):
            self.images.append(np.array(sitk.GetArrayFromImage(sitk.ReadImage(path_to_mha[i]))))

        mkdir_p(saving_path)
        plt.set_cmap('gray')
        while how_many_from_one > 0:
            image_to_save = np.zeros((5,
                                      216,
                                      160))
            rand_value = rnd.randint(30, len(self.images[0]) - 30)
            for i in range(0, len(path_to_mha)):
                try:
                    image_to_save[i] = self.images[i][rand_value]
                except:
                    print('ahi')
                    print(self.images[i][rand_value].shape)
                    print(type(self.images))
                    print(type(self.images))
                    print('*')
                    continue
            print(image_to_save.shape)
            image_to_save = image_to_save.reshape((216 * 5, 160))
            print(image_to_save.shape)
            # image_to_save = resize(image_to_save, (5*216, 160), mode='constant')
            # image_to_save = image_to_save.resize(5*216, 160)
            plt.imsave(saving_path + str(global_counter) + '.png',
                       image_to_save)
            global_counter += 1
            how_many_from_one -= 1


if __name__ == '__main__':

    all_patients_path = glob('/Users/Cesare/Desktop/lavoro/cnn_med3d/BRATS-2/Image_Data/HG/**')

    global_counter = 0
    how_many_from_patient = 4
    for i in range(0, 10):
        patient_number = rnd.randint(0, len(all_patients_path) - 1)
        # print(glob(all_patients_path[patient_number] + '/**/**.mha'))
        single_patient_path_mod = glob(all_patients_path[patient_number] + '/**/**.mha')
        ImagePngConverter(global_counter, path_to_mha=single_patient_path_mod, how_many_from_one=how_many_from_patient)
        global_counter += how_many_from_patient
