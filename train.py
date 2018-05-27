from model import init_models, BATCH_SIZE, IMG_SIZE, LD_CROP_SIZE, MAX_LEN, MIN_LEN
from data_generator import gen_batch
from utils import ElapsedTimer

import numpy as np
import os, h5py

def trainC(Cmodel, batch,epoch):
    origins, complnet_inputs, holed_origins, masks, _ = batch
    mse_loss = Cmodel.train_on_batch([holed_origins, complnet_inputs, masks], 
                                     origins)
    #print('epoch %d: [C mse loss: %e]' % (epoch, mse_loss))
    return mse_loss

VALIDS = np.ones((BATCH_SIZE, 1))
FAKES = np.zeros((BATCH_SIZE, 1))
def trainD(Cmodel, Dmodel, batch, epoch):
    origins, complnet_inputs, holed_origins, masks, ld_crop_yxhws = batch
    completed = Cmodel.predict([holed_origins, complnet_inputs, masks])
    d_loss_real = Dmodel.train_on_batch([origins,ld_crop_yxhws], VALIDS)
    d_loss_fake = Dmodel.train_on_batch([completed,ld_crop_yxhws], FAKES)
    bce_d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    #print('epoch %d: [D bce loss: %e]' % (epoch, bce_d_loss))
    return bce_d_loss

def trainC_in(joint_model, batch, epoch):
    origins, complnet_inputs, holed_origins, masks, ld_crop_yxhws = batch
    joint_loss,mse,gan = joint_model.train_on_batch([holed_origins, complnet_inputs, masks,
                                                     ld_crop_yxhws],
                                                    [origins, VALIDS])
    #print('epoch %d: [joint loss: %e | mse loss: %e, gan loss: %e]' % (epoch, joint_loss, mse, gan))
    return joint_loss,mse,gan

def log_and_save(C_loss,D_loss,CD_loss, 
                 epoch,batch, 
                 num_epoch,tc,td, save_interval):
    mse_loss = C_loss
    bce_d_loss = D_loss
    joint_loss,mse,gan = CD_loss
    if epoch < tc:
        print('epoch %d: [C mse loss: %e]' % (epoch, mse_loss))
    else:
        print('epoch %d: [C mse loss: %e] [D bce loss: %e]' 
                % (epoch, mse_loss, bce_d_loss))
        if epoch >= tc + td:
            print('epoch %d: [joint loss: %e | mse loss: %e, gan loss: %e]' 
                    % (epoch, joint_loss, mse, gan))

            if epoch % save_interval == 0 or epoch == num_epoch - 1:
                result_dir = 'output'
                completed = Cmodel.predict([masked_origins, 
                                            complnet_inputs, 
                                            masks])
                np.save(os.path.join(result_dir,'I_O_GT__%d.npy' % epoch),
                        np.array([complnet_inputs,completed,origins]))
                Cmodel.save_weights(os.path.join(result_dir, 
                                                 "complnet_%d.h5" % epoch))
                Dmodel.save_weights(os.path.join(result_dir, 
                                                 "discrimnet_%d.h5" % epoch))

Cmodel, Dmodel, CDmodel = init_models()

from concurrent.futures import ProcessPoolExecutor
#data_file = h5py.File('./data/mini_data.h5','r') 
data_file = h5py.File('./data/data128.h5','r') 
#-------------------------------------------------------------------------------
data_arr = data_file['images']
mean_pixel_value = data_file['mean_pixel_value'][()] / 255

#save_interval = 20
save_interval = 2
#num_epoch = 240
#tc = int(num_epoch * 0.18)
#td = int(num_epoch * 0.02)
num_epoch = 13 # 
tc = 2 # 2
td = 1
print('num_epoch=',num_epoch,'tc=',tc,'td=',td)

timer = ElapsedTimer('Total Training')
for epoch in range(num_epoch):
    epoch_timer = ElapsedTimer('1 epoch training time')
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
    epoch_timer.elapsed_time()

    if epoch < tc:
        print('epoch %d: [C mse loss: %e]' % (epoch, mse_loss))
    else:
        print('epoch %d: [D bce loss: %e]' % (epoch, bce_d_loss))
        if epoch >= tc + td:
            print('epoch %d: [joint loss: %e | mse loss: %e, gan loss: %e]' 
                    % (epoch, joint_loss, mse, gan))
    #origins, complnet_inputs, masked_origins, masks, ld_crop_yxhws = batch
    #log_and_save(mse_loss, bce_d_loss, (joint_loss,mse,gan),
                 #epoch, batch, num_epoch,tc,td, save_interval)
    '''
    if epoch < tc:
        print('epoch %d: [C mse loss: %e]' % (epoch, mse_loss))
    else:
        print('epoch %d: [C mse loss: %e] [D bce loss: %e]' 
                % (epoch, mse_loss, bce_d_loss))
        if epoch >= tc + td:
            print('epoch %d: [joint loss: %e | mse loss: %e, gan loss: %e]' 
                    % (epoch, joint_loss, mse, gan))

            if epoch % save_interval == 0 or epoch == num_epoch - 1:
                result_dir = 'output'
                completed = Cmodel.predict([masked_origins, 
                                            complnet_inputs, 
                                            masks])
                np.save(os.path.join(result_dir,'I_O_GT__%d.npy' % epoch),
                        np.array([complnet_inputs,completed,origins]))
                Cmodel.save_weights(os.path.join(result_dir, 
                                                 "complnet_%d.h5" % epoch))
                Dmodel.save_weights(os.path.join(result_dir, 
                                                 "discrimnet_%d.h5" % epoch))
    '''
#-------------------------------------------------------------------------------
time_str = timer.elapsed_time()
data_file.close()
#import mailing
#mailing.send_mail_to_kur(time_str)
'''
if __name__ == "__main__":
    timer = ElapsedTimer()
    main()
    timer.elapsed_time()

    #plot_model(Cmodel, to_file='mse_model.png', show_shapes=True)
    #model.summary()
'''

