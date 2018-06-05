from model import init_models, BATCH_SIZE, IMG_SIZE, LD_CROP_SIZE, MAX_LEN, MIN_LEN
from data_generator import gen_batch
from utils import ElapsedTimer

import numpy as np
import os, h5py

def trainC(Cmodel, batch,epoch):
    origins, complnet_inputs, holed_origins, masks, _ = batch
    mse_loss = Cmodel.train_on_batch([holed_origins, complnet_inputs, masks], 
                                     origins)
    return mse_loss

VALIDS = np.ones((BATCH_SIZE, 1))
FAKES = np.zeros((BATCH_SIZE, 1))
def trainD(Cmodel, Dmodel, batch, epoch):
    origins, complnet_inputs, holed_origins, masks, ld_crop_yxhws = batch
    completed = Cmodel.predict([holed_origins, complnet_inputs, masks])
    d_loss_real = Dmodel.train_on_batch([origins,ld_crop_yxhws], VALIDS)
    d_loss_fake = Dmodel.train_on_batch([completed,ld_crop_yxhws], FAKES)
    bce_d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    return bce_d_loss

def trainC_in(joint_model, batch, epoch):
    origins, complnet_inputs, holed_origins, masks, ld_crop_yxhws = batch
    joint_loss,mse,gan = joint_model.train_on_batch([holed_origins, complnet_inputs, masks,
                                                     ld_crop_yxhws],
                                                    [origins, VALIDS])
    return joint_loss,mse,gan

def save(Cmodel,Dmodel,batch, period,epoch,num_epoch, result_dir):
    origins, complnet_inputs, holed_origins, masks, ld_crop_yxhws = batch
    if epoch % period == 0 or epoch == num_epoch - 1:
        completed = Cmodel.predict([holed_origins, complnet_inputs, masks])

        np.save(os.path.join(result_dir,'I_O_GT__%d.npy' % epoch),
                np.array([complnet_inputs,completed,origins]))

        Cmodel.save_weights(
            os.path.join(result_dir, "complnet_%d.h5" % epoch)
        )
        Dmodel.save_weights(
            os.path.join(result_dir, "discrimnet_%d.h5" % epoch)
        )

Cmodel, Dmodel, CDmodel = init_models()

save_interval = 50
num_epoch = 1100
tc = int(num_epoch * 0.18)
td = int(num_epoch * 0.02)
'''
save_interval = 2
num_epoch = 3 # 
tc = 1 # 2
td = 1
'''
dataset_path = './data/gray2_128sqr_3crop_32batch.h5'
print('num_epoch=',num_epoch,'tc=',tc,'td=',td)
print('on dataset: ',dataset_path)

data_file = h5py.File(dataset_path,'r') 
#-------------------------------------------------------------------------------
data_arr = data_file['images'] # already preprocessed, float32.
mean_pixel_value = data_file['mean_pixel_value'][()] # value is float

timer = ElapsedTimer('Total Training')
#-------------------------------------------------------------------------------
for epoch in range(num_epoch):
    #epoch_timer = ElapsedTimer('1 epoch training time')
    #--------------------------------------------------------------------------
    for batch in gen_batch(data_arr, BATCH_SIZE, IMG_SIZE, LD_CROP_SIZE,
                           MIN_LEN, MAX_LEN, mean_pixel_value):
        if epoch < tc:
            mse_loss = trainC(Cmodel, batch, epoch)
        else:
            bce_d_loss = trainD(Cmodel, Dmodel, batch, epoch)
            if epoch >= tc + td:
                joint_loss,mse,gan = trainC_in(CDmodel, batch, epoch)
    #--------------------------------------------------------------------------
    #epoch_timer.elapsed_time()

    if epoch < tc:
        print('epoch %d: [C mse loss: %e]' % (epoch, mse_loss))
    else:
        if epoch >= tc + td:
            print('epoch %d: [joint loss: %e | mse loss: %e, gan loss: %e]' 
                    % (epoch, joint_loss, mse, gan))
        else:
            print('epoch %d: [D bce loss: %e]' % (epoch, bce_d_loss))
    save(Cmodel,Dmodel,batch, save_interval,epoch,num_epoch, 'output')
#-------------------------------------------------------------------------------
time_str = timer.elapsed_time()
data_file.close()

import mailing
mailing.send_mail_to_kur('Training Ended Successfully!', time_str)

