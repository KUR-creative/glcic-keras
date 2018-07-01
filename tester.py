#import time
import numpy as np
import h5py, cv2, os

def write_result_img(npy_path,img_path,
                     batch_size,size,num_channels=3):
    result = np.load(npy_path) 
    print(result.shape)
    # size = image size
    result_img = np.empty((batch_size*size, 3*size, num_channels))
    for i in range(batch_size): 
        result_img[i*size:(i+1)*size, 0*size:1*size] = result[0,i]
        result_img[i*size:(i+1)*size, 1*size:2*size] = result[1,i]
        result_img[i*size:(i+1)*size, 2*size:3*size] = result[2,i]
    # convert correct image format
    result_img = (result_img * 255).astype(np.uint8)
    if num_channels != 1:
        result_img = cv2.cvtColor(result_img,cv2.COLOR_BGR2RGB)
    cv2.imwrite(img_path, result_img)

#TODO: load saved complnet and predict!
#TODO: create interactive demo!

                
if __name__ == "__main__":
    bat_size = 64#32#96
    img_size = 128
    write_result_img('./output/I_O_GT__19.npy',
                     './output/result19.png',bat_size,img_size)
    #for i in range(40,180+20,20):
    #for i in range(20,500,20):
    for i in range(0,18+1,2):
    #for i in range(60,220+1,20):
        write_result_img('./output/I_O_GT__%d.npy' % i,
                         './output/result%d.png' % i,
                         bat_size,img_size,1)
        print(i)
    '''
    #unittest.main()
    batch_size = 32
    img_size = 192
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
                     './output/result.png',bat_size,img_size,1)
    write_result_img('./output/I_O_GT__160.npy',
                     './output/result12.png',bat_size,img_size,1)
    write_result_img('./output/I_O_GT__199.npy',
                     './output/result199.png',bat_size,img_size,1)
    '''
    '''
    '''

'''
# chunk_generator is for hdf5 file generation!!!!
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
'''
