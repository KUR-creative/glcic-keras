from layers import completion_net, discrimination_net, cropping

from keras.layers import Input, Add, Multiply, merge
from keras.models import Model
from keras.engine.topology import Container
from keras.optimizers import Adadelta
from keras.utils import plot_model

import numpy as np

BATCH_SIZE = False
IMG_SIZE = False

# LD means Local Discrimnator
LD_CROP_SIZE = False
HOLE_MIN_LEN = False
HOLE_MAX_LEN = False

IMG_SHAPE = False
LD_CROP_SHAPE = False

VAR_IMG_SHAPE = False
MASK_SHAPE = False
VAR_MASK_SHAPE = False

ALPHA = False
D_MODEL_LR = False

def set_global_consts(batch_size, img_size, 
                      is_grayscale, ld_crop_size_proportion,
                      rand_hole_min_proportion,rand_hole_max_proportion,
                      alpha, d_model_lr):
    global BATCH_SIZE, IMG_SIZE
    global IMG_SHAPE, LD_CROP_SHAPE, VAR_IMG_SHAPE, MASK_SHAPE, VAR_MASK_SHAPE
    global LD_CROP_SIZE, HOLE_MIN_LEN, HOLE_MAX_LEN
    global ALPHA, D_MODEL_LR

    BATCH_SIZE = batch_size
    IMG_SIZE = img_size

    LD_CROP_SIZE = int(IMG_SIZE * ld_crop_size_proportion)  
    HOLE_MAX_LEN = int(LD_CROP_SIZE * rand_hole_max_proportion)
    HOLE_MIN_LEN = int(LD_CROP_SIZE * rand_hole_min_proportion)

    num_channels = 1 if is_grayscale else 3

    IMG_SHAPE = (IMG_SIZE,IMG_SIZE,num_channels)
    LD_CROP_SHAPE = (LD_CROP_SIZE,LD_CROP_SIZE,num_channels)
    VAR_IMG_SHAPE = (None,None,num_channels)

    MASK_SHAPE = (IMG_SIZE,IMG_SIZE,1)
    VAR_MASK_SHAPE = (None,None,1)

    ALPHA = alpha
    D_MODEL_LR = d_model_lr


def init_models(Cnet_path=None, Dnet_path=None):
    assert ((Cnet_path == None) ^ (Dnet_path == None)) is False
    #----------------------------- completion_model ----------------------------
    holed_origins_inp = Input(shape=IMG_SHAPE, name='holed_origins_inp')
    complnet_inp = Input(shape=IMG_SHAPE, name='complnet_inp')
    masks_inp = Input(shape=MASK_SHAPE, name='masks_inp')

    complnet_out = completion_net(VAR_IMG_SHAPE)(complnet_inp)
    merged_out = Add()([holed_origins_inp, 
                         Multiply()([complnet_out, 
                                     masks_inp])])
    compl_model = Model([holed_origins_inp, complnet_inp, masks_inp], 
                        merged_out)
    if Cnet_path: 
        compl_model.load_weights(Cnet_path, by_name=True)
    compl_model.compile(loss='mse', optimizer=Adadelta())

    #compl_model.summary()
    #plot_model(compl_model, to_file='C_model.png', show_shapes=True)

    #--------------------------- discrimination_model --------------------------
    origins_inp = Input(shape=IMG_SHAPE, name='origins_inp')
    crop_yxhw_inp = Input(shape=(4,), dtype=np.int32, name='yxhw_inp')

    local_cropped = merge([origins_inp,crop_yxhw_inp], mode=cropping, 
                          output_shape=MASK_SHAPE, name='local_crop')
    discrim_out = discrimination_net(IMG_SHAPE, LD_CROP_SHAPE)([origins_inp, 
                                                                local_cropped])
    discrim_model = Model([origins_inp,crop_yxhw_inp], discrim_out)
    if Dnet_path: 
        discrim_model.load_weights(Dnet_path, by_name=True)
    discrim_model.compile(loss='binary_crossentropy', 
                          optimizer=Adadelta(lr=D_MODEL_LR)) # good? lol
                          #optimizer=Adam(lr=0.000001))

    #discrim_model.summary()
    #plot_model(discrim_model, to_file='D_model.png', show_shapes=True)

    #------------------------------- joint_model -------------------------------
    d_container = Container([origins_inp,crop_yxhw_inp], 
                            discrim_out, name='D_container')
    d_container.trainable = False
    joint_model = Model([holed_origins_inp, complnet_inp, masks_inp, 
                         crop_yxhw_inp],
                        [merged_out, d_container([merged_out,crop_yxhw_inp])])

    joint_model.compile(loss=['mse', 'binary_crossentropy'],
                        loss_weights=[1.0, ALPHA], optimizer=Adadelta())

    #joint_model.summary()
    #plot_model(joint_model, to_file='joint_model.png', show_shapes=True)

    return compl_model, discrim_model, joint_model

