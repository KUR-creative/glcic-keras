import numpy as np
import h5py, cv2, os

def random_masked(mat, min_len,max_len, mask_val=1,bg_val=0): 
    '''           ^^^
    WARNING: It modifies mat!
    
    If you don't want side-effects, pass tuple(shape of matrix) 
    as mat, then this function create new matrix and return it.

    min/max_len: min/max length of mask
    => if mat.shape < max_len: it's ok!

    return
     mat, (top left coord of mask), (height,width of mask)

     If input matrix is not square, it would be problematic..
     but I don't know why..
    '''
    if type(mat) is tuple:
        shape = mat
        mat = np.zeros(shape,dtype=np.uint8)
    mat[:] = bg_val
    mask_h,mask_w = np.random.randint(min_len, max_len+1, 
                                      dtype=int, size=2)
    mat_h,mat_w = mat.shape[:2]
    max_y = mat_h - min_len + 1 
    max_x = mat_w - min_len + 1 

    # top left coord of mask.
    y = np.random.randint(0,max_y, dtype=int)
    x = np.random.randint(0,max_x, dtype=int)

    mat[y:y+mask_h, x:x+mask_w] = mask_val
    return mat, (y,x), (mask_h,mask_w)

def get_ld_crop_yx(originHW, localHW, maskYX, maskHW):
    '''
    height,width of original image
    height,width of local crop 
    left top coordinate(Y,X) of mask
    height,width of mask

    return y,x: coordinate of local crop
    '''
    oH,oW = originHW
    lH,lW = localHW
    mY,mX = maskYX
    mH,mW = maskHW

    half_lH, half_lW = lH // 2, lW // 2
    center_mY, center_mX = mY + mH // 2, mX + mW // 2

    #threshold of center of mask
    minY, minX = half_lH, half_lW
    maxY, maxX = oH - half_lH, oW - half_lW

    if   center_mX > maxX: center_mX = maxX
    elif center_mX < minX: center_mX = minX
    if   center_mY > maxY: center_mY = maxY
    elif center_mY < minY: center_mY = minY

    y,x = center_mY - half_lH, center_mX - half_lW
    return y,x

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

def gen_batch(data_arr, batch_size, img_shape, ld_crop_size,
              min_mask_len, max_mask_len, mean_pixel_value):
    img_size = img_shape[0]
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

    #print(data_arr.shape)
    #cv2.imshow('org',data_arr[1]); cv2.waitKey(0)
    for i in range(0,arr_len, batch_size):
        if i + batch_size > arr_len: #TODO: => or > ?
            break
        origins = np.empty((batch_size,) + img_shape, dtype=data_arr.dtype)
        #print(origins.shape)
        for n in range(batch_size):
            idx = idxes[i:i+batch_size][n]
            origins[n] = data_arr[idx]
            #print(type(idx))
            #cv2.imshow('org',data_arr[idx]); cv2.waitKey(0)
            #cv2.imshow('org',origins[n]); cv2.waitKey(0)

        #cv2.imshow('wtf',origins[1]); cv2.waitKey(0)

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

'''
import unittest
class Test(unittest.TestCase):
    pass

if __name__ == "__main__":
    unittest.main()

from model import init_models, BATCH_SIZE, IMG_SIZE, LD_CROP_SIZE, MAX_LEN, MIN_LEN
data_file = h5py.File('./data/test.h5','r') 
data_arr = data_file['images'] # already preprocessed, float32.
mean_pixel_value = data_file['mean_pixel_value'][()] # value is float
for batch in gen_batch(data_arr, BATCH_SIZE, IMG_SIZE, LD_CROP_SIZE,
                       MIN_LEN, MAX_LEN, mean_pixel_value):
    origins, complnet_inputs, holed_origins, masks, ld_crop_yxhws = batch
    cv2.imshow('origin',origins[1]); cv2.waitKey(0)
data_file.close()
'''
