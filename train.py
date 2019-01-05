from image_generator import generator
import csv
import cv2
import os
from sklearn.model_selection import train_test_split 
from models import TransferVGG, Nvidia
import sys


samples = []

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
#     print('Error: driving_log.csv is not existed.')
#     sys.exit(0)

print('amount of sample -> {}'.format(len(samples)))


# Trimmed image format 
height, width, ch = 160, 320, 3
input_size = 100

# model = TransferVGG(128, 32)
model = Nvidia()

save_path = './train_models/model3.h5'

# Train the model
batch_size =256
epochs = 10


if os.path.basename(save_path) in os.listdir('./train_models'):
    model.load(save_path)
    

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=batch_size, data_augmentation=True, is_exist_log=is_exist_log)
validation_generator = generator(validation_samples, batch_size=batch_size, is_exist_log=is_exist_log)
try:
    model.train(train_generator, validation_generator, epochs, len(train_samples)*6, len(validation_samples), batch_size)
except KeyboardInterrupt:
    model.save(save_path)
    sys.exit(0)

model.save(save_path)
model.plot_error()



