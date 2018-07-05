'''
import sys
imgpath = sys.argv[1]
origin, hw = load_image(imgpath)
mean_mask, not_mask = mask_from_user(hw, origin)
'''
from data_generator import gen_batch
from layers import completion_net, discrimination_net
from tester_ui import tester_ui
import numpy as np
import cv2
from keras.layers import Input, Add, Multiply, merge
from keras.models import Model
from keras.utils import plot_model

def mse(A,B):
    return ((A-B)**2).mean()
def normalized(uint8img):
    return uint8img.astype(np.float32) / 255
def inverse_normalized(float32img):
    return (float32img * 255).astype(np.uint8)

def load_compl_model(model_path, img_shape=(None,None,3)):
    complnet_inp = Input(shape=img_shape, name='complnet_inp')
    complnet_out = completion_net(img_shape)(complnet_inp)
    compl_model = Model([complnet_inp], complnet_out)
    compl_model.load_weights(model_path, by_name=True)

    '''
    compl_model.summary()
    plot_model(compl_model, to_file='C_model_test.png', 
               show_shapes=True)
    '''
    return compl_model

def load_image(imgpath):
    origin = cv2.imread(imgpath)
    origin = cv2.cvtColor(origin,cv2.COLOR_BGR2RGB)
    origin = normalized(origin)
    return origin

def mask_from_user(mask_hw, origin):
    h,w = mask_hw
    print('-------- ui start! --------')
    bgr_origin = cv2.cvtColor(origin,cv2.COLOR_RGB2BGR)
    mask = tester_ui(bgr_origin)
    mean_mask = mask * np.mean(origin) # images 2
    return mean_mask, np.logical_not(mean_mask).astype(np.float32)

kernel = np.ones((1,1),np.uint8)
def load_mask_pair(imgpath, origin_mean_pixel_value, 
                   mask_channel=0, threshold=0.1):
    '''
    ex) ' ':hole, 'm':mean pixel value of origin
        origin          mask         not_mask
    12345678901234|              |11111111111111
    923759 9237 22|      m    m  |111111 1111 11
    93298  927  32|     mm   mm  |11111  111  11
    2398       239|    mmmmmmm   |1111       111
    2397492  49272|       mm     |1111111  11111
    28394   347927|     mmm      |11111   111111
    85729547328492|              |11111111111111
    '''
    mask = load_image(imgpath)
    mask = (mask[:,:,mask_channel] > threshold).astype(np.uint8)
    mask = cv2.dilate(mask,kernel,iterations=1)
    mean_mask = mask.astype(np.float32) * origin_mean_pixel_value # images 2
    not_mask = np.logical_not(mean_mask).astype(np.float32)
    return mean_mask, not_mask
    
def padding_removed(padded_img,no_pad_shape):
    '''
    pH,pW,_ = padded_img.shape
    nH,nW,_ = no_pad_shape
    dH = pH - nH
    dW = pW - nW
    # TODO: change this! it's temporary implementation!
    # TODO: 0~pH-dH is incorrect!
    return padded_img[0:pH-dH,0:pW-dW]
    '''
    return adjusted_image(padded_img, no_pad_shape)

def completion(completion_model, origin, mean_mask, not_mask):
    h,w = origin.shape[:2]
    holed_origin = origin * not_mask

    cnet_input = np.copy(holed_origin) + mean_mask
    cnet_input = cnet_input[:,:,0].reshape((1,h,w,1))

    cnet_output = completion_model.predict( [cnet_input] )
    cnet_output = cnet_output.reshape(cnet_output.shape[1:])
    cnet_output = padding_removed(cnet_output,origin.shape)

    mask = np.logical_not(not_mask).astype(np.float32)
    return cnet_output * mask + holed_origin

def adjusted_image(image, shape, pad_value=0): # tested on only grayscale image.
    h,w,_ = image.shape

    d_h = shape[0] - h
    if d_h > 0:
        d_top = d_h // 2
        d_bot = d_h - d_top
        image = np.pad(image, [(d_top,d_bot),(0,0),(0,0)], 
                       mode='constant', constant_values=pad_value)
        #print('+ y',image.shape)
    else:
        d_top = abs(d_h) // 2
        d_bot = abs(d_h) - d_top
        image = image[d_top:h-d_bot,:]
        #print('- y',image.shape)

    d_w = shape[1] - w
    if d_w > 0:
        d_left = d_w // 2
        d_right = d_w - d_left
        image = np.pad(image, [(0,0),(d_left,d_right),(0,0)],
                       mode='constant', constant_values=pad_value)
        #print('+ x',image.shape)
    else:
        d_left = abs(d_w) // 2
        d_right = abs(d_w) - d_left
        image = image[:,d_left:w-d_right]
        #print('- x',image.shape)
    return image

def mse_ratio_similarity(actual_img, expected_img, max_err_img):
    result_mse = mse(expected_img, actual_img)
    max_mse = mse(expected_img, max_err_img)
    similarity = (max_mse - result_mse) / max_mse
    return similarity

import unittest
class Test_adjusted_image(unittest.TestCase):
    def assert_adjustment(self, src_shape, dst_shape, visual_check=False):
        h,w,_ = src_shape
        src = np.arange(h*w, dtype=np.uint8).reshape(src_shape)
        adjusted = adjusted_image(src, dst_shape)
        self.assertEqual(adjusted.shape, dst_shape)
        if visual_check:
            cv2.imshow('src', src); cv2.waitKey(0)
            cv2.imshow('adjusted', adjusted); cv2.waitKey(0)

    def test_identity_case(self):
        shape = (100,100,1)
        src = np.arange(10000,dtype=np.uint8).reshape((100,100,1))
        adjusted = adjusted_image(src,shape)
        self.assertTrue( np.array_equal(adjusted,src) )
        self.assertEqual( adjusted.shape, shape )         # (0,0)

    def test_shrinking_case(self):
        print('---shrinking---')
        self.assert_adjustment( (200,100,1),(100,100,1) ) # (-,0)
        self.assert_adjustment( (100,200,1),(100,100,1) ) # (0,-)
        self.assert_adjustment( (200,200,1),(100,100,1) ) # (-,-)

    def test_padding_case(self):
        print('---padding---')
        self.assert_adjustment( (100,200,1),(200,200,1) ) # (+,0)
        self.assert_adjustment( (200,100,1),(200,200,1) ) # (0,+)
        print('---- now! ----')
        self.assert_adjustment( (100,100,1),(200,200,1) ) # (+,+)

    def test_mixed_case(self):
        self.assert_adjustment( (100,200,1),(200,100,1) ) # (+,-)
        self.assert_adjustment( (200,100,1),(100,200,1) ) # (-,+)

    def test_pad_val(self):
        shape = (200,200,1)
        src = np.zeros(10000,dtype=np.float32).reshape((100,100,1))
        adjusted = adjusted_image(src,shape,1)
        self.assertNotEqual( np.sum(src), np.sum(adjusted) )
        self.assertEqual( adjusted.shape, shape )  
        

def main():
    # score: origin, mask, answer
    # completed_image: model, origin, mask
    origin_path = './eval-data/mini_evals/001_clean.png'
    mask_path = './eval-data/mini_evals/008_mask.png'

    origin = load_image(origin_path)
    mean_mask, not_mask = load_mask_pair(mask_path, 
                                         np.mean(origin))

    h,w = origin.shape[:2]
    m_h, m_w = mean_mask.shape[:2]
    origin = origin[:,:,0].reshape((h,w,1)) # grayscale only!
    mean_mask = adjusted_image( mean_mask.reshape([m_h,m_w,1]), (h,w,1) )
    not_mask = adjusted_image( not_mask.reshape([m_h,m_w,1]), (h,w,1), 1.0 )
    #cv2.imshow('not_mask',not_mask); cv2.waitKey(0)
    #cv2.imshow('mean mask',mean_mask); cv2.waitKey(0)
    #cv2.imshow('not mask',mean_mask); cv2.waitKey(0)

    compl_model = load_compl_model('./old_complnets/complnet_5.h5',
    #compl_model = load_compl_model('./output/complnet_0.h5',
    #compl_model = load_compl_model('./old_complnets/complnet_499.h5',
    #compl_model = load_compl_model('./old_complnets/complnet_9000.h5',
    #compl_model = load_compl_model('./old_complnets/192x_200e_complnet_199.h5',
    #compl_model = load_compl_model('./old_complnets/192x_200e_complnet_190.h5',
                                   (None,None,1))
    completed = completion(compl_model,
                           origin, mean_mask, not_mask)

    #bgr_origin = cv2.cvtColor(origin,cv2.COLOR_RGB2BGR)
    cv2.imshow('origin',origin); cv2.waitKey(0)#-----------------
    cv2.imshow('mean_mask',mean_mask); cv2.waitKey(0)
    cv2.imshow('not_mask',not_mask); cv2.waitKey(0)
    #cv2.imshow('mask',mask); cv2.waitKey(0)#---------------------
    #cv2.imshow('holed_origin',holed_origin); cv2.waitKey(0)
    #cv2.imshow('complnet_input',complnet_input); cv2.waitKey(0)
    #cv2.imshow('complnet_output',complnet_output); cv2.waitKey(0)
    #completed = cv2.cvtColor(completed,cv2.COLOR_RGB2BGR)
    cv2.imshow('completed',completed); cv2.waitKey(0)#-----------

    #print(origin.shape); #print(mask.shape); #print('is it ok?')

    answer = load_image(origin_path) # answer
    cv2.imshow('answer',answer); cv2.waitKey(0)
    mask = np.logical_not(not_mask).astype(np.float32)
    expected = answer * mask

    answer_uint8 = inverse_normalized(answer)
    max_err_img = cv2.bitwise_not(answer_uint8)
    cv2.imshow('max error img',max_err_img); cv2.waitKey(0)
    #print(np.sum(max_err_img))

    max_err_img = normalized(max_err_img) * mask
    #cv2.imshow('masked max error img',max_err_img); cv2.waitKey(0)
    #print(np.sum(max_err_img))
    #print(answer.shape)
    #max_err_img = cv2.bitwise_not(answer.astype()).astype(np.float32) * mask
    actual = completed * mask

    #cv2.imshow('expected',expected); cv2.waitKey(0)
    #cv2.imshow('actual',actual); cv2.waitKey(0)
    #cv2.imshow('max error img',max_err_img); cv2.waitKey(0)

    '''
    result_mse = mse(expected,actual)
    max_mse = mse(expected,max_err_img)
    print('mse =', result_mse)
    print('max mse =', max_mse)
    print('similarity = {:3f}%'.format((max_mse - result_mse) / max_mse * 100))
    print('error = {:3f}%'.format(100 - (max_mse - result_mse) / max_mse * 100) )
    '''
    sim = mse_ratio_similarity(actual, expected, max_err_img)
    err = 1 - sim
    from skimage.measure import compare_ssim
    masked_ssim = compare_ssim(expected[:,:,0],actual[:,:,0]) # inputs must be 2D array!
    full_ssim = compare_ssim(answer[:,:,0],completed[:,:,0]) # inputs must be 2D array!

    print('masked mse ratio similarity = {:3f}'.format(sim))
    print('masked mse ratio error = {:3f}'.format(err))
    print('masked ssim = {}'.format(masked_ssim))
    print('full ssim = {}'.format(full_ssim))

if __name__ == '__main__':
    #unittest.main()
    main()
