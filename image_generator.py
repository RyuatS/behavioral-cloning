import cv2
import sklearn
import numpy as np
import random
import os


def generator(samples, batch_size=32, data_augmentation=False, is_exist_log=False):
    """
    this is the function for creating generators.
    if data_augmentation is True, flip the image through to increase the image.

    return => (image_array, angles_array)
    """
    num_samples = len(samples)

    while 1:
        random.shuffle(samples)

        if data_augmentation:
            for offset in range(0, num_samples, int(batch_size//6)):
                batch_samples = samples[offset:offset+batch_size]

                images = []
                angles = []
                for batch_sample in batch_samples:
                    center_basename = batch_sample[0].split('/')[-1]
                    left_basename = batch_sample[1].split('/')[-1]
                    right_basename = batch_sample[2].split('/')[-1]
                    if is_exist_log:
                        center_filename = '../../IMG/' + center_basename
                        left_filename = '../../IMG/' + left_basename
                        right_filename = '../../IMG/' + right_basename
                    else:
                        center_filename = './data/IMG/' + center_basename
                        left_filename = './data/IMG/' + left_basename
                        right_filename = './data/IMG/' + right_basename
                    # read image
                    center_image = cv2.imread(center_filename)
                    left_image = cv2.imread(left_filename)
                    right_image = cv2.imread(right_filename)
                    # convert color
                    center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                    right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                    # flip image
                    center_image_flipped = cv2.flip(center_image, 1)
                    left_image_flipped = cv2.flip(left_image, 1)
                    right_image_flipped = cv2.flip(right_image, 1)

                    # calc angle
                    center_angle = float(batch_sample[3])
                    center_angle_flipped = -center_angle
                    correction = 0.2
                    left_angle = center_angle + correction
                    left_angle_flipped = -left_angle
                    right_angle = center_angle - correction
                    right_angle_flipped = -right_angle


                    images.append(center_image)
                    images.append(center_image_flipped)
                    images.append(left_image)
                    images.append(left_image_flipped)
                    images.append(right_image)
                    images.append(right_image_flipped)

                    angles.append(center_angle)
                    angles.append(center_angle_flipped)
                    angles.append(left_angle)
                    angles.append(left_angle_flipped)
                    angles.append(right_angle)
                    angles.append(right_angle_flipped)

                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)

        else:
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset+batch_size]
                images = []
                angles = []
                for batch_sample in batch_samples:
                    basename = batch_sample[0].split('/')[-1]
                    if is_exist_log:
                        filename = '../../IMG/' + basename
                    else:
                        filename = './data/IMG/' + basename
                    center_image = cv2.imread(filename)
                    center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                    center_angle = float(batch_sample[3])
                    images.append(center_image)
                    angles.append(center_angle)


                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)
