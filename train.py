from image_generator import generator
import csv
import cv2
import os
from sklearn.model_selection import train_test_split
from models import TransferVGG, Nvidia
import sys

# This array has the each row in csv file.
samples = []

# Does the own collection data exist?
# exist -> True, not exist -> False.
is_exist_log = 'driving_log.csv' in os.listdir('//home')


if is_exist_log:
    print('load the my data')
    with open('../../driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            samples.append(line)

else:
    print('load the template data')
    with open('./data/driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        for line in reader:
            samples.append(line)

# print amount of the data.
print('amount of sample -> {}'.format(len(samples)))


# if you want to use the transfer-learning model. comment in below.
# model = TransferVGG(128, 32)

# use Nvidia model from model.py.
model = Nvidia()

# file_path for saving model's weights.
save_path = './train_models/model.h5'

# if file_path is exist in train_models directory, load the file.
if os.path.basename(save_path) in os.listdir('./train_models'):
    model.load(save_path)

# split the data set to training and validation data.
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# hyper parameter
batch_size = 32
epochs = 10

# create generator
train_generator = generator(train_samples, batch_size=batch_size, data_augmentation=True, is_exist_log=is_exist_log)
validation_generator = generator(validation_samples, batch_size=batch_size, is_exist_log=is_exist_log)


try:
    # train the model
    model.train(train_generator, validation_generator, epochs, len(train_samples)*6, len(validation_samples), batch_size)
except KeyboardInterrupt:
    model.save(save_path)
    sys.exit(0)

# save the model
model.save(save_path)
model.plot_error()
