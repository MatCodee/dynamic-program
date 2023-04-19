'''
Este es le modelo de la red neironal donde vamos a ejecutar los parametros necesarios
para que el modelo genere un input de exito
'''
import numpy as np
from tensorflow.keras.layers import Input

from InputVideo import reading_video

# Este es el procesamiento de los datos
def processInput():
    video_array = reading_video()
    print(video_array)

    input_shape = (None, None, 3)  # Formato de imagen RGB
    input_a = Input(shape=input_shape)  # Video sin editar
    input_b = Input(shape=input_shape)  # Video editado de referencia

    input_data = (input_a, input_b)

    
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten, Dropout
from keras.layers import Reshape, Conv2DTranspose
from keras.layers import BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Model
from keras.utils import to_categorical, plot_model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


# latend_dim: es la dimension del vector

class CGAN: 
    def __init__(self,img_width,img_height,n_channels,n_classes):
        self.img_width = img_width
        self.img_height = img_height
        
        # no entieno lo que es n_channels (numero de canales?)
        self.n_channels = n_channels
        self.img_shape = (self.img_width,self.img_height,self.n_channels)

        # no entiendo lo que es el numero de clases
        self.n_classes = n_classes
        self.latent_dim = 100
        
        # Este es el optimizador es una variable creada durante el constructor
        #optimizer = Adam(0.0002, 0.5)

        # Construir el compilador del discrimindador


    def build_discriminator_model(self):
        model = keras.Sequential()
        model.add(Conv2D(128, (3,3), strides=(2,2), padding='same',input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1,activation='sigmoid'))

        # compiler model:
        opt = Adam(lr=0.0002,beta_1=0.5)
        model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
        return model


    def build_generator(self,latent_dim):
        model = keras.Sequential()
        n_nodes = 128 * 8 * 8
        model.add(Dense(n_nodes,input_dim=latent_dim)) 
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((8,8,128))) # Esto es porque esta comenzando con 128 x 8 x 8

        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(3, (8,8), activation='tanh', padding='same'))
        return model


    def define_gan(self,generator,discriminator):
        discriminator.trainable = False # definimos el discriminador como no entrenable

        #  conectamos el generador con el con el discriminador
        model = keras.Sequential()
        model.add(generator)
        model.add(discriminator)

        opt = Adam(lr=0.0002,beta_1=0.5)
        model.compile(loss='binary_crossentropy',optimizer=opt)
        return model

    def train(self):
        pass