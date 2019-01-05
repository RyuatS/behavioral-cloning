from keras.layers import Input, Lambda, Cropping2D, Dense, Flatten, GlobalAveragePooling2D, Conv2D, Dropout
from keras.models import Model
from keras.callbacks import TensorBoard
import tensorflow as tf
import matplotlib.pyplot as plt 
from keras import optimizers
import sys


class MyModel:
    
    def train(self, generator, val_generator, epochs, num_train, num_val, batch_size):
        tensorboard_callback = TensorBoard(log_dir='./logs/', write_graph=True, write_images=True)
        self.history = self.model.fit_generator(generator=generator, steps_per_epoch=num_train//batch_size,\
                                                validation_data=val_generator, validation_steps=num_val//batch_size,\
                                                epochs=epochs, verbose=1, callbacks=[tensorboard_callback])
   

    

    def save(self, file_path):
        print('saving model .....')
        try:
            self.model.save_weights(file_path)
        except:
            print('save error')
            
   
    def load(self, file_path):
        print('loading model.....')
        self.model.load_weights(file_path)
  
        
    def predict(self, input_img, batch_size):
        return self.model.predict(input_img, batch_size)
        
    def plot_error(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validaiton_set'], loc='upper right')
        plt.savefig('./model_mean_squared_error.png')
        
        

class TransferVGG(MyModel):
    def __init__(self, fc1_nodes = 128, fc2_nodes=32):
        from keras.applications.vgg16 import VGG16
        self.input_size = 100
        self.input_shape = (160, 320, 3)
        vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(self.input_size, self.input_size, 3))
        # Load pre-training model ( VGG16)
        for _ in range(8):
            vgg16.layers.pop()

        # Define the model
        input_img = Input(shape=self.input_shape)
        img_cropped = Cropping2D(cropping=((50, 20), (0, 0)))(input_img)
        img_resized = Lambda(lambda image: tf.image.resize_images(image, (self.input_size, self.input_size)))(img_cropped)
        img_normalized = Lambda(lambda x: (x / 255.0) - 0.5)(img_resized)
        vgg = vgg16(img_normalized)
        gpa = GlobalAveragePooling2D()(vgg)
        fc1 = Dense(fc1_nodes)(gpa)
        fc2 = Dense(fc2_nodes)(fc1)
        predictions = Dense(1)(fc1)

        # Create the model
        self.model = Model(inputs=input_img, outputs=predictions)
        # Compile the model
        self.model.compile(optimizer='adam', loss='mse')
  

    def freeze(self):
        self.model.layers[5].trainabel = False

        
class Nvidia(MyModel):
    def __init__(self):
        self.input_shape = (160, 320, 3)
        
        # Define the model 
        input_img = Input(shape=self.input_shape)
        img_cropped = Cropping2D(cropping=((50, 20), (0, 0)))(input_img)
        img_resized = Lambda(lambda image: tf.image.resize_images(image, (66, 200)))(img_cropped)
        img_normalized = Lambda(lambda x: (x / 255.0) - 0.5)(img_resized)
        conv2d_1 = Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), padding='valid')(img_normalized)
        conv2d_2 = Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), padding='valid')(conv2d_1)
        conv2d_3 = Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), padding='valid')(conv2d_2)
        conv2d_4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid')(conv2d_3)
        conv2d_5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid')(conv2d_4)
        flatten = Flatten()(conv2d_5)
        fc1 = Dense(100, activation='relu')(flatten)
        drop_fc1 = Dropout(0.5)(fc1)
        fc2 = Dense(50, activation='relu')(drop_fc1)
        drop_fc2 = Dropout(0.5)(fc2)
        fc3 = Dense(10, activation='relu')(drop_fc2)
        predictions = Dense(1)(fc3)
        
        # Create the model
        self.model = Model(inputs=input_img, outputs=predictions)
        self.model.compile(optimizer='adam', loss='mse')
    

    