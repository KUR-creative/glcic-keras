from layers import completion_net, discrimination_net, cropping

from keras.layers import Input, Add, Multiply, merge
from keras.models import Model
from keras.engine.topology import Container
from keras.optimizers import Adadelta, Adam
from data_generator import gen_batch
from utils import ElapsedTimer
from keras.utils import plot_model

import numpy as np
import os, h5py
np.set_printoptions(threshold=np.nan, linewidth=np.nan)


#BATCH_SIZE = 16#16 64 occur OOM 
BATCH_SIZE = 32#16 64 occur OOM 
IMG_SIZE = 128
LD_CROP_SIZE = IMG_SIZE // 2  # LD means Local Discrimnator
#MAX_LEN = IMG_SIZE // 2
#MIN_LEN = IMG_SIZE // 4
MAX_LEN = IMG_SIZE // 3
MIN_LEN = IMG_SIZE // 5

IMG_SHAPE = (IMG_SIZE,IMG_SIZE,3)
LD_CROP_SHAPE = (LD_CROP_SIZE,LD_CROP_SIZE,3)
MASK_SHAPE = (IMG_SIZE,IMG_SIZE,1)
VAR_IMG_SHAPE = (None,None,3)
VAR_MASK_SHAPE = (None,None,1)

#def completion_model():
complnet_inp = Input(shape=IMG_SHAPE, name='complnet_inp')
holed_origins_inp = Input(shape=IMG_SHAPE, name='holed_origins_inp')
masks_inp = Input(shape=MASK_SHAPE, name='masks_inp')

complnet_out = completion_net(VAR_IMG_SHAPE)(complnet_inp)
merged_out = Add()([holed_origins_inp, 
                     Multiply()([complnet_out, 
                                 masks_inp])])
compl_model = Model([holed_origins_inp, 
                     complnet_inp, 
                     masks_inp], merged_out)
compl_model.compile(loss='mse', optimizer=Adadelta())

#def discrimination_model():
origins_inp = Input(shape=IMG_SHAPE, name='origins_inp')
crop_yxhw_inp = Input(shape=(4,), dtype=np.int32, name='yxhw_inp')
local_cropped = merge([origins_inp,crop_yxhw_inp], mode=cropping, 
                      output_shape=MASK_SHAPE, name='local_crop')
discrim_out = discrimination_net(IMG_SHAPE,
                                 LD_CROP_SHAPE)([origins_inp,
                                                 local_cropped])
discrim_model = Model([origins_inp,crop_yxhw_inp], discrim_out)
discrim_model.compile(loss='binary_crossentropy', 
                      optimizer=Adadelta(lr=0.01)) # good? lol
                      #optimizer=Adam(lr=0.000001))
discrim_model.summary()
plot_model(discrim_model, to_file='D_model.png', show_shapes=True)
                      
#def joint_model():
d_container = Container([origins_inp,crop_yxhw_inp], discrim_out,
                        name='D_container')
d_container.trainable = False
joint_model = Model([holed_origins_inp,complnet_inp,masks_inp,
                     crop_yxhw_inp],
                    [merged_out,
                     d_container([merged_out,crop_yxhw_inp])])

alpha = 0.0004
joint_model.compile(loss=['mse', 'binary_crossentropy'],
                    loss_weights=[1.0, alpha], optimizer=Adadelta())
joint_model.summary()
plot_model(joint_model, to_file='joint_model.png', show_shapes=True)
timer = ElapsedTimer('Total Training')

from concurrent.futures import ProcessPoolExecutor
#data_file = h5py.File('./data/mini_data.h5','r') 
data_file = h5py.File('./data/data128.h5','r') 
#--------------------------------------------------------------------------------------
data_arr = data_file['images']
mean_pixel_value = data_file['mean_pixel_value'][()] / 255
def generate_batch(data_arr):
    batch = gen_batch(data_arr, BATCH_SIZE, IMG_SIZE, LD_CROP_SIZE,
                      MIN_LEN, MAX_LEN, mean_pixel_value)
    yield batch
#save_interval = 20
save_interval = 2
#num_epoch = 240
#tc = int(num_epoch * 0.18)
#td = int(num_epoch * 0.02)
num_epoch = 6 # 
tc = 2 # 2
td = 1
print('num_epoch=',num_epoch,'tc=',tc,'td=',td)

batch_stream = None
#with ProcessPoolExecutor(max_workers=4) as exe:
    #batch_stream = exe.map(generate_batch, data_arr)

def training(epoch,batch):
    origins, complnet_inputs, masked_origins, masks, ld_crop_yxhws = batch

    #batch_timer = ElapsedTimer('1 batch training time')
    #--------------------------------------------------------------------------------------
    if epoch < tc:
        mse_loss = compl_model.train_on_batch([masked_origins, complnet_inputs, masks],
                                              origins)
    else:
        completed = compl_model.predict([masked_origins, complnet_inputs, masks])
        d_loss_real = discrim_model.train_on_batch([origins,ld_crop_yxhws],valids)
        d_loss_fake = discrim_model.train_on_batch([completed,ld_crop_yxhws],fakes)
        bce_d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        if epoch >= tc + td:
            joint_loss,mse,gan = joint_model.train_on_batch([masked_origins,complnet_inputs,masks,
                                                             ld_crop_yxhws],
                                                            [origins, valids])

valids = np.ones((BATCH_SIZE, 1))
fakes = np.zeros((BATCH_SIZE, 1))
for epoch in range(num_epoch):
    epoch_timer = ElapsedTimer('1 epoch training time')
    #--------------------------------------------------------------------------------------
    for batch in gen_batch(data_arr, BATCH_SIZE, IMG_SIZE, LD_CROP_SIZE,
                      MIN_LEN, MAX_LEN, mean_pixel_value):
        '''
        training(epoch,batch)
        '''
        origins, complnet_inputs, masked_origins, masks, ld_crop_yxhws = batch

        #batch_timer = ElapsedTimer('1 batch training time')
        #--------------------------------------------------------------------------------------
        if epoch < tc:
            mse_loss = compl_model.train_on_batch([masked_origins, complnet_inputs, masks],
                                                  origins)
        else:
            completed = compl_model.predict([masked_origins, complnet_inputs, masks])
            d_loss_real = discrim_model.train_on_batch([origins,ld_crop_yxhws],valids)
            d_loss_fake = discrim_model.train_on_batch([completed,ld_crop_yxhws],fakes)
            bce_d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            if epoch >= tc + td:
                joint_loss,mse,gan = joint_model.train_on_batch([masked_origins,complnet_inputs,masks,
                                                                 ld_crop_yxhws],
                                                                [origins, valids])
        #--------------------------------------------------------------------------------------
        #batch_timer.elapsed_time()
    #--------------------------------------------------------------------------------------
    epoch_timer.elapsed_time()

    if epoch < tc:
        print('epoch %d: [C mse loss: %e]' % (epoch, mse_loss))
    else:
        print('epoch %d: [C mse loss: %e] [D bce loss: %e]' % (epoch, mse_loss, bce_d_loss))
        if epoch >= tc + td:
            print('epoch %d: [joint loss: %e | mse loss: %e, gan loss: %e]' % (epoch, joint_loss, mse,gan))

            if epoch % save_interval == 0 or epoch == num_epoch - 1:
                result_dir = 'output'
                completed = compl_model.predict([masked_origins, 
                                                 complnet_inputs, 
                                                 masks])
                np.save(os.path.join(result_dir,'I_O_GT__%d.npy' % epoch),
                        np.array([complnet_inputs,completed,origins]))
                        # save predicted image of last batch in epoch.
                compl_model.save_weights(os.path.join(result_dir, 
                                                      "complnet_%d.h5" % epoch))
                discrim_model.save_weights(os.path.join(result_dir, 
                                                        "discrimnet_%d.h5" % epoch))
#--------------------------------------------------------------------------------------
time_str = timer.elapsed_time()
data_file.close()
#import mailing
#mailing.send_mail_to_kur(time_str)
'''
if __name__ == "__main__":
    timer = ElapsedTimer()
    main()
    timer.elapsed_time()

    #plot_model(compl_model, to_file='mse_model.png', show_shapes=True)
    #model.summary()
'''

