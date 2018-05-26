#import time
import numpy as np
from utils import *
import h5py, cv2, os
import matplotlib.pyplot as plt

import time
class ElapsedTimer(object):
    def __init__(self,string='Elapsed'):
        self.start_time = time.time()
        self.string = string

    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print(self.string + ": %s " % self.elapsed(time.time() - self.start_time),
              flush=True)
        return (self.string + ": %s " % self.elapsed(time.time() - self.start_time))

def get_random_maskeds(batch_size, img_size, 
                       min_mask_len, max_mask_len):
    maskeds = np.empty((batch_size,img_size,img_size,1))
    mask_yxhws = []
    for idx in range(batch_size):
        _, mask_yx, mask_hw = random_masked(maskeds[idx], 
                                            min_mask_len,
                                            max_mask_len)
        y,x = mask_yx
        h,w = mask_hw
        mask_yxhws.append((y,x,h,w))
    return maskeds, mask_yxhws

def get_complnet_inputs(masked_origins,mask_yxhws, 
                        batch_size, mean_pixel_value):
    complnet_inputs = np.copy(masked_origins)
    for idx in range(batch_size):
        y,x,h,w = mask_yxhws[idx]
        complnet_inputs[idx][y:y+h,x:x+w] = mean_pixel_value
    return complnet_inputs

def gen_batch(data_arr, batch_size, img_size, ld_crop_size,
              min_mask_len, max_mask_len, mean_pixel_value):
    def _get_crop_yx(mask_yxhw_arr):
        mY,mX, mH,mW = mask_yxhw_arr
        y,x = get_ld_crop_yx((img_size,img_size),
                             (ld_crop_size,ld_crop_size),
                             (mY,mX), (mH,mW))
        return y,x
    ''' yield minibatches '''
    arr_len = data_arr.shape[0] // batch_size # never use remainders..

    idxes = np.arange(arr_len,dtype=np.uint32)
    np.random.shuffle(idxes) #shuffle needed.

    for i in range(0,arr_len, batch_size):
        if i + batch_size > arr_len: #TODO: => or > ?
            break
        unpreprocessed_imgs = np.empty((batch_size,
                                        img_size,img_size,3),
                                       dtype=np.uint8)
        for n in range(batch_size):
            idx = idxes[i:i+batch_size][n]
            unpreprocessed_imgs[n] = data_arr[idx]

        origins = unpreprocessed_imgs.astype(np.float32) / 255

        maskeds, mask_yxhws = get_random_maskeds(batch_size, 
                                                 img_size, 
                                                 min_mask_len, 
                                                 max_mask_len);

        not_maskeds = np.logical_not(maskeds).astype(np.float32)
        masked_origins = origins * not_maskeds

        complnet_inputs = get_complnet_inputs(masked_origins,
                                              mask_yxhws,
                                              batch_size,
                                              mean_pixel_value)
        ld_crop_yxhws = np.empty((batch_size,4),dtype=int)
        for idx,(y,x) in enumerate(map(_get_crop_yx,mask_yxhws)):
            ld_crop_yxhws[idx] = y,x, ld_crop_size,ld_crop_size
        
        yield origins, complnet_inputs, masked_origins, maskeds, ld_crop_yxhws

def write_result_img(npy_path,img_path,batch_size,size):
    result = np.load(npy_path) 
    # size = image size
    result_img = np.empty((batch_size*size, 3*size, 3))
    for i in range(batch_size): 
        result_img[i*size:(i+1)*size, 0*size:1*size] = result[0,i]
        result_img[i*size:(i+1)*size, 1*size:2*size] = result[1,i]
        result_img[i*size:(i+1)*size, 2*size:3*size] = result[2,i]
    # convert correct image format
    result_img = (result_img * 255).astype(np.uint8)
    result_img = cv2.cvtColor(result_img,cv2.COLOR_BGR2RGB)
    cv2.imwrite(img_path, result_img)

#TODO: load saved complnet and predict!
#TODO: create interactive demo!

def chunk_generator(np_array,chk_size):
    length = len(np_array)
    for beg_idx in range(0,length, chk_size):
        yield np_array[beg_idx:beg_idx+chk_size]

def iter_mean(prev_mean,prev_size, now_sum,now_size):
    total = prev_size + now_size
    return prev_mean*prev_size/total + now_sum/total

import unittest
class Test_chunk_generator(unittest.TestCase):
    def test_empty(self):
        arr = list(chunk_generator([],100))
        self.assertEqual(arr,[])

    def test_array_size_is_divisible_by_chunk_size(self):
        num_chks = 10
        arr = []
        for chunk in chunk_generator(np.ones(num_chks*10),
                                     num_chks):
            arr.append(chunk)
        self.assertEqual(len(arr), num_chks)

    def test_array_size_is_not_divisible_by_chunk_size(self):
        num_chks = 10
        chk_size = 9
        remainder_size = 2
        length = remainder_size + num_chks*chk_size
        src_arr = [1] * (remainder_size + num_chks*chk_size)
        dst_arr = [0] * (remainder_size + num_chks*chk_size)
        arr = []
        for idx,chunk in enumerate(chunk_generator(src_arr, 
                                                   chk_size)):
            beg_idx = idx*chk_size
            dst_arr[beg_idx:beg_idx+chk_size] = chunk
            arr.append(chunk)
        self.assertEqual(len(dst_arr), length)
        #self.assertEqual(len(dst_arr[-1]), remainder_size)
        print(dst_arr)
        print(len(dst_arr))
        print(arr)

    #@unittest.skip('later')
    def test_chunks_indexing(self):
        chk_size = 100
        num_chks = 10
        remainder_size = 42
        length = num_chks*chk_size + remainder_size

        src_arr = np.ones(length)
        dst_arr = np.empty(length)
        for idx,chunk in enumerate(chunk_generator(src_arr, 
                                                   chk_size)):
            now_chk_size = chunk.shape[0] # it would be smaller than chk_size!
            print(now_chk_size)
            beg_idx = idx*chk_size
            dst_arr[beg_idx:beg_idx+now_chk_size] = chunk
        self.assertEqual(dst_arr.shape[0], length)
        #self.assertEqual(arr[-1].shape[0], remainder_size)
        print(dst_arr)
                
if __name__ == "__main__":
    '''
    #unittest.main()
    batch_size = 16
    img_size = 128
    maxl = img_size // 2
    minl = img_size // 4
    with h5py.File('./data128_half.h5','r') as data_file:
        data_arr = data_file['images']
        mean_pixel_value = data_file['mean_pixel_value'][()] / 255

        for batch in gen_batch(data_arr, batch_size, 
                               img_size, img_size // 2, 
                               minl,maxl,mean_pixel_value):
            origins, complnet_inputs, masked_origins, maskeds, ld_crop_yxhws = batch
            lY,lX, lH,lW = ld_crop_yxhws[0]
            #print('uwang good',ld_crop_yxhws)
            cv2.imshow('img',origins[0]); cv2.waitKey(0)
            #cv2.imshow('img2',origins[batch_size-1]); cv2.waitKey(0)
            cv2.imshow('ab',masked_origins[0]); cv2.waitKey(0) 
            #cv2.imshow('ab2',masked_origins[batch_size-1]); cv2.waitKey(0)
            cv2.imshow('complnet_inp',complnet_inputs[0]); cv2.waitKey(0)
            #cv2.imshow('complnet_inp2',complnet_inputs[batch_size-1]); cv2.waitKey(0) 
            cv2.imshow('ld_crop',complnet_inputs[0][lY:lY+lH,lX:lX+lW]); cv2.waitKey(0)
            #cv2.imshow('ld_crop2',complnet_inputs[batch_size-1][lY:lY+lH,lX:lX+lW]); cv2.waitKey(0)

    write_result_img('./output/I_O_GT__180.npy',
                     './output/result.png',bat_size,img_size)
    write_result_img('./output/I_O_GT__160.npy',
                     './output/result12.png',bat_size,img_size)
    write_result_img('./output/I_O_GT__199.npy',
                     './output/result199.png',bat_size,img_size)
    '''
    bat_size = 32
    img_size = 128
    #for i in range(40,180+20,20):
    for i in range(4,12+1,2):
    #for i in range(60,220+1,20):
        write_result_img('./output/I_O_GT__%d.npy' % i,
                         './output/result%d.png' % i,
                         bat_size,img_size)
        print(i)
    write_result_img('./output/I_O_GT__239.npy',
                     './output/result239.png',bat_size,img_size)
