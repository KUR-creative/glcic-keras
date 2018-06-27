from keras.layers import Flatten, Activation, Conv2D, Conv2DTranspose, Dense, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
import tensorflow as tf

def cropping(imgs_yxhws):
    def crop(img_yxhw):
        img,yxhw = img_yxhw
        y = yxhw[0]; x = yxhw[1]
        h = yxhw[2]; w = yxhw[3]
        return tf.image.crop_to_bounding_box(img, y,x, h,w)
    return tf.map_fn(crop, imgs_yxhws, dtype=tf.float32, infer_shape=False)
 
def add_layer_BN_relu(model,layer_fn,*args,**kargs):
    model.add(layer_fn(*args,**kargs))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

def set_layer_BN_relu(input,layer_fn,*args,**kargs):
    x = layer_fn(*args,**kargs)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def completion_net(input_shape=(256, 256, 3),name='complnet'):
    model = Sequential(name=name)
    '''
    # default strides = (1,1)
    #--------------------------------
    add_layer_BN_relu(model, Conv2D,  64, (5,5), padding='same', input_shape=input_shape)
    #----------------
    add_layer_BN_relu(model, Conv2D, 128, (3,3), strides=(2,2), padding='same')
    add_layer_BN_relu(model, Conv2D, 128, (3,3), padding='same')
    #--------
    add_layer_BN_relu(model, Conv2D, 256, (3,3), strides=(2,2), padding='same')
    add_layer_BN_relu(model, Conv2D, 256, (3,3), padding='same')
    add_layer_BN_relu(model, Conv2D, 256, (3,3), padding='same')
    # dilation
    add_layer_BN_relu(model, Conv2D, 256, (3,3), dilation_rate=( 2, 2), padding='same')
    add_layer_BN_relu(model, Conv2D, 256, (3,3), dilation_rate=( 4, 4), padding='same')
    add_layer_BN_relu(model, Conv2D, 256, (3,3), dilation_rate=( 8, 8), padding='same')
    add_layer_BN_relu(model, Conv2D, 256, (3,3), dilation_rate=(16,16), padding='same')
    #--------
    add_layer_BN_relu(model, Conv2D, 256, (3,3), padding='same')
    add_layer_BN_relu(model, Conv2D, 256, (3,3), padding='same')
    #----------------
    add_layer_BN_relu(model, Conv2DTranspose, 128, (4,4), strides=(2,2), padding='same')
    add_layer_BN_relu(model, Conv2D, 128, (3,3), padding='same')
    #--------------------------------
    add_layer_BN_relu(model, Conv2DTranspose, 64, (4,4), strides=(2,2), padding='same')
    add_layer_BN_relu(model, Conv2D,  32, (3,3), padding='same')

    model.add(Conv2D( 3, (3,3), padding='same')) # no BN!
    model.add(Activation('sigmoid'))
    '''
    # default strides = (1,1)
    #--------------------------------
    add_layer_BN_relu(model, Conv2D,  32, (5,5), padding='same', input_shape=input_shape)
    #----------------
    add_layer_BN_relu(model, Conv2D,  64, (3,3), strides=(2,2), padding='same')
    add_layer_BN_relu(model, Conv2D,  64, (3,3), padding='same')
    #--------
    add_layer_BN_relu(model, Conv2D, 128, (3,3), strides=(2,2), padding='same')
    add_layer_BN_relu(model, Conv2D, 128, (3,3), padding='same')
    add_layer_BN_relu(model, Conv2D, 128, (3,3), padding='same')
    # dilation
    add_layer_BN_relu(model, Conv2D, 128, (3,3), dilation_rate=( 2, 2), padding='same')
    add_layer_BN_relu(model, Conv2D, 128, (3,3), dilation_rate=( 4, 4), padding='same')
    add_layer_BN_relu(model, Conv2D, 128, (3,3), dilation_rate=( 8, 8), padding='same')
    add_layer_BN_relu(model, Conv2D, 128, (3,3), dilation_rate=(16,16), padding='same')
    #--------
    add_layer_BN_relu(model, Conv2D, 128, (3,3), padding='same')
    add_layer_BN_relu(model, Conv2D, 128, (3,3), padding='same')
    #----------------
    add_layer_BN_relu(model, Conv2DTranspose, 128, (4,4), strides=(2,2), padding='same')
    add_layer_BN_relu(model, Conv2D,  64, (3,3), padding='same')
    #--------------------------------
    add_layer_BN_relu(model, Conv2DTranspose, 64, (4,4), strides=(2,2), padding='same')
    add_layer_BN_relu(model, Conv2D,  32, (3,3), padding='same')

    model.add(Conv2D( 1, (3,3), padding='same')) # no BN!
    model.add(Activation('sigmoid'))
    return model

def discrimination_net(global_shape=(128, 128, 3), local_shape=(64, 64, 3),name='discrimnet'):
    g_img = Input(shape=global_shape)
    '''
    x_g = set_layer_BN_relu(g_img, Conv2D,  64, (5,5), strides=(2,2), padding='same')
    x_g = set_layer_BN_relu(  x_g, Conv2D, 128, (5,5), strides=(2,2), padding='same')
    x_g = set_layer_BN_relu(  x_g, Conv2D, 256, (5,5), strides=(2,2), padding='same')
    x_g = Flatten()(x_g)
    x_g = Dense(1024)(x_g) 

    l_img = Input(shape=local_shape)
    x_l = set_layer_BN_relu(l_img, Conv2D,  64, (5,5), strides=(2,2), padding='same')
    x_l = set_layer_BN_relu(  x_l, Conv2D, 128, (5,5), strides=(2,2), padding='same')
    x_l = Flatten()(x_l)
    x_l = Dense(512)(x_l) 

    x = concatenate([x_g, x_l]) # no activation. or..?
    x = Dense(1, activation='sigmoid')(x)
    '''
    x_g = set_layer_BN_relu(g_img, Conv2D,  32, (5,5), strides=(2,2), padding='same')
    x_g = set_layer_BN_relu(  x_g, Conv2D,  64, (5,5), strides=(2,2), padding='same')
    x_g = set_layer_BN_relu(  x_g, Conv2D, 128, (5,5), strides=(2,2), padding='same')
    x_g = set_layer_BN_relu(  x_g, Conv2D, 256, (5,5), strides=(2,2), padding='same')
    x_g = set_layer_BN_relu(  x_g, Conv2D, 256, (5,5), strides=(2,2), padding='same')
    x_g = set_layer_BN_relu(  x_g, Conv2D, 256, (5,5), strides=(2,2), padding='same')
    x_g = Flatten()(x_g)
    x_g = Dense(512)(x_g) 

    l_img = Input(shape=local_shape)
    x_l = set_layer_BN_relu(l_img, Conv2D,  32, (5,5), strides=(2,2), padding='same')
    x_l = set_layer_BN_relu(  x_l, Conv2D,  64, (5,5), strides=(2,2), padding='same')
    x_l = set_layer_BN_relu(  x_l, Conv2D, 128, (5,5), strides=(2,2), padding='same')
    x_l = set_layer_BN_relu(  x_l, Conv2D, 256, (5,5), strides=(2,2), padding='same')
    x_l = set_layer_BN_relu(  x_l, Conv2D, 256, (5,5), strides=(2,2), padding='same')
    x_l = Flatten()(x_l)
    x_l = Dense(512)(x_l) 

    x = concatenate([x_g, x_l]) # no activation. or..?
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=[g_img, l_img], outputs=x, name=name)


import numpy as np
import keras.backend as K
def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

if __name__ == "__main__":
    from keras.utils import plot_model
    complnet = completion_net()
    complnet.summary()
    plot_model(complnet, to_file='Cnet.png', show_shapes=True)
    discrimnet = discrimination_net()
    discrimnet.summary()
    plot_model(discrimnet, to_file='Dnet.png', show_shapes=True)

    print(get_model_memory_usage(6,complnet),'GB')
    print(get_model_memory_usage(6,discrimnet),'GB')
