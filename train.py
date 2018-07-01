from model import init_models, BATCH_SIZE, IMG_SHAPE, LD_CROP_SIZE, HOLE_MAX_LEN, HOLE_MIN_LEN
from data_generator import gen_batch
from utils import ElapsedTimer

from tqdm import tqdm
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

from tqdm import tqdm
def train(DATASET_NAME, NUM_EPOCH, Tc, Td, SAVE_INTERVAL, MAILING_ENABLED,learned_data_ratio):
    data_file = h5py.File(DATASET_NAME,'r') 
    #-------------------------------------------------------------------------------
    data_arr = data_file['images'] # already preprocessed, float32.
    mean_pixel_value = data_file['mean_pixel_value'][()] # value is float
    learned_arr_len = int(data_arr.shape[0] * learned_data_ratio)
    learned_arr_len = learned_arr_len - (learned_arr_len % BATCH_SIZE)#never use remainders..
    print('data_arr shape: ', data_arr.shape)
    print('length of data to learn: ', learned_arr_len)

    timer = ElapsedTimer('Total Training')
    #-------------------------------------------------------------------------------
    for epoch in tqdm(range(NUM_EPOCH)):
        #epoch_timer = ElapsedTimer()
        #--------------------------------------------------------------------------
        for batch in gen_batch(data_arr, BATCH_SIZE, IMG_SHAPE, LD_CROP_SIZE,
                               HOLE_MIN_LEN, HOLE_MAX_LEN, mean_pixel_value,
                               learned_arr_len):
            if epoch < Tc:
                mse_loss = trainC(Cmodel, batch, epoch)
            else:
                bce_d_loss = trainD(Cmodel, Dmodel, batch, epoch)
                if epoch >= Tc + Td:
                    joint_loss,mse,gan = trainC_in(CDmodel, batch, epoch)
        #--------------------------------------------------------------------------
        if epoch < Tc:
            print('epoch {}: [C mse loss: {}]'.format(epoch, mse_loss), flush=True)#, end='')
        else:
            if epoch >= Tc + Td:
                print('epoch {}: [joint loss: {} | mse loss: {}, gan loss: {}]'\
                       .format(epoch, joint_loss, mse, gan), flush=True)#, end='')
            else:
                print('epoch {}: [D bce loss: {}]'.format(epoch, bce_d_loss), flush=True)#, end='')
        #epoch_timer.elapsed_time()
        #print()
        save(Cmodel,Dmodel,batch, SAVE_INTERVAL,epoch,NUM_EPOCH, 'output')
    #-------------------------------------------------------------------------------
    time_str = timer.elapsed_time()
    data_file.close()

    if MAILING_ENABLED:
        import mailing
        mailing.send_mail_to_kur(time_str)

if __name__ == "__main__":
    DATASET_NAME = './data/test.h5'
    NUM_EPOCH = 5#1100
    Tc = 1#int(NUM_EPOCH * 0.18)
    Td = 1#int(NUM_EPOCH * 0.02)
    SAVE_INTERVAL = 2#50
    MAILING_ENABLED = False
    train(DATASET_NAME, NUM_EPOCH, Tc, Td, SAVE_INTERVAL, MAILING_ENABLED)
