from layers import completion_net, discrimination_net, cropping

from keras.layers import Input, Add, Multiply, merge
from keras.models import Model
from keras.engine.topology import Container
from keras.optimizers import Adadelta
from keras.utils import plot_model

import numpy as np

#BATCH_SIZE = 16#16 64 occur OOM 
#BATCH_SIZE = 32#16 64 occur OOM 
BATCH_SIZE = 128
IMG_SIZE = 128
LD_CROP_SIZE = IMG_SIZE // 2  # LD means Local Discrimnator
#MAX_LEN = IMG_SIZE // 2
#MIN_LEN = IMG_SIZE // 4
MAX_LEN = IMG_SIZE // 3
MIN_LEN = IMG_SIZE // 5

#IMG_SHAPE = (IMG_SIZE,IMG_SIZE,3)
#LD_CROP_SHAPE = (LD_CROP_SIZE,LD_CROP_SIZE,3)
#VAR_IMG_SHAPE = (None,None,3)
#VAR_MASK_SHAPE = (None,None,1)
IMG_SHAPE = (IMG_SIZE,IMG_SIZE,1)
LD_CROP_SHAPE = (LD_CROP_SIZE,LD_CROP_SIZE,1)
MASK_SHAPE = (IMG_SIZE,IMG_SIZE,1)
VAR_IMG_SHAPE = (None,None,1)
VAR_MASK_SHAPE = (None,None,1)


alpha = 0.0004
Dmodel_lr = 0.01

def init_models():
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
    compl_model.compile(loss='mse', optimizer=Adadelta())

    #--------------------------- discrimination_model --------------------------
    origins_inp = Input(shape=IMG_SHAPE, name='origins_inp')
    crop_yxhw_inp = Input(shape=(4,), dtype=np.int32, name='yxhw_inp')

    local_cropped = merge([origins_inp,crop_yxhw_inp], mode=cropping, 
                          output_shape=MASK_SHAPE, name='local_crop')
    discrim_out = discrimination_net(IMG_SHAPE, LD_CROP_SHAPE)([origins_inp, 
                                                                local_cropped])
    discrim_model = Model([origins_inp,crop_yxhw_inp], 
                          discrim_out)
    discrim_model.compile(loss='binary_crossentropy', 
                          optimizer=Adadelta(lr=Dmodel_lr)) # good? lol
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
                        loss_weights=[1.0, alpha], optimizer=Adadelta())

    #joint_model.summary()
    #plot_model(joint_model, to_file='joint_model.png', show_shapes=True)
    return compl_model, discrim_model, joint_model

