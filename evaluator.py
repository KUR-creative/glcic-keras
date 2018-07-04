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

def load_image(img_path):
    origin = cv2.imread(img_path)
    origin = cv2.cvtColor(origin,cv2.COLOR_BGR2RGB)
    origin = origin.astype(np.float32) / 255
    return origin, origin.shape[:2]

def mask_from_user(mask_hw, origin):
    h,w = mask_hw
    print('-------- ui start! --------')
    bgr_origin = cv2.cvtColor(origin,cv2.COLOR_RGB2BGR)
    mask = tester_ui(bgr_origin)
    mean_mask = mask * np.mean(origin) # images 2
    return mean_mask, np.logical_not(mean_mask).astype(np.float32)

kernel = np.ones((1,1),np.uint8)
def load_r_mask(imgpath, origin):
    mask, hw = load_image(imgpath)

    mask = (mask[:,:,0] > 0.1).astype(np.uint8)#.astype(np.float32)
    #cv2.imshow('mask',mask.astype(np.float32)); cv2.waitKey(0)
    mask = cv2.dilate(mask,kernel,iterations=1)
    #cv2.imshow('mask',mask.astype(np.float32)); cv2.waitKey(0)
    mask = mask.astype(np.float32)

    mean_mask = mask * np.mean(origin) # images 2
    #cv2.imshow('mask',mask); cv2.waitKey(0)
    #cv2.imshow('mean mask',mean_mask); cv2.waitKey(0)
    return mean_mask, np.logical_not(mean_mask).astype(np.float32)
    
def padding_removed(padded_img,no_pad_shape):
    pH,pW,_ = padded_img.shape
    nH,nW,_ = no_pad_shape
    dH = pH - nH
    dW = pW - nW
    # TODO: change this! it's temporary implementation!
    # TODO: 0~pH-dH is incorrect!
    return padded_img[0:pH-dH,0:pW-dW]

def adjusted_image(image, shape): # tested on only grayscale image.
    h,w,_ = image.shape
    adjusted = image

    d_h = shape[0] - image.shape[0] 
    if d_h > 0:
        d_top = d_h // 2; d_bot = d_h - d_top
        adjusted = np.pad(adjusted, [(d_top,d_bot), (0,0), (0,0)], 
                          mode='constant')
        #print('+ y',adjusted.shape)
    else:
        d_h = abs(d_h)
        d_top = d_h // 2; d_bot = d_h - d_top
        adjusted = adjusted[d_top:h-d_bot,:]
        #print('- y',adjusted.shape)

    d_w = shape[1] - image.shape[1]
    if d_w > 0:
        d_left = d_w // 2; d_right = d_w - d_left
        adjusted = np.pad(adjusted, [(0,0), (d_left,d_right), (0,0)], 
                          mode='constant')
        #print('+ x',adjusted.shape)
    else:
        d_w = abs(d_w)
        d_left = d_w // 2; d_right = d_w - d_left
        adjusted = adjusted[:,d_left:w-d_right]
        #print('- x',adjusted.shape)
    return adjusted

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


def main():
    img_no = '011'
    origin, hw = load_image('./eval-data/mini_evals/'+img_no+'.jpg')
    mean_mask, not_mask = load_r_mask('./eval-data/mini_evals/'+img_no+'_mask.png',
                                      origin)
    h,w = hw
    origin = origin[:,:,0].reshape((h,w,1)) # grayscale only!
    mean_mask = mean_mask.reshape((h,w,1))
    not_mask = not_mask.reshape((h,w,1))
    #print(mean_mask.shape, not_mask.shape)

    holed_origin = origin * not_mask
    complnet_input = np.copy(holed_origin) + mean_mask

    complnet_input = complnet_input[:,:,0]
    complnet_input = complnet_input.reshape((1,h,w,1))

    #compl_model = load_compl_model('./old_complnets/complnet_5.h5',
    #compl_model = load_compl_model('./output/complnet_0.h5',
    #compl_model = load_compl_model('./old_complnets/complnet_499.h5',
    compl_model = load_compl_model('./old_complnets/complnet_9000.h5',
    #compl_model = load_compl_model('./old_complnets/192x_200e_complnet_199.h5',
                                   (None,None,1))
    complnet_output = compl_model.predict(
                        [complnet_input.reshape((1,h,w,1))]
                      )
    complnet_output = complnet_output.reshape(
                        complnet_output.shape[1:]
                      )
    complnet_output = padding_removed(complnet_output, origin.shape)

    mask = np.logical_not(not_mask).astype(np.float32)
    completed = complnet_output * mask + holed_origin


    #bgr_origin = cv2.cvtColor(origin,cv2.COLOR_RGB2BGR)
    cv2.imshow('origin',origin); cv2.waitKey(0)#-----------------
    #cv2.imshow('mean_mask',mean_mask); cv2.waitKey(0)
    #cv2.imshow('not_mask',not_mask); cv2.waitKey(0)
    cv2.imshow('mask',mask); cv2.waitKey(0)#---------------------
    #cv2.imshow('holed_origin',holed_origin); cv2.waitKey(0)
    #cv2.imshow('complnet_input',complnet_input); cv2.waitKey(0)
    #cv2.imshow('complnet_output',complnet_output); cv2.waitKey(0)
    #completed = cv2.cvtColor(completed,cv2.COLOR_RGB2BGR)
    cv2.imshow('completed',completed); cv2.waitKey(0)#-----------

    #print(origin.shape); #print(mask.shape); #print('is it ok?')

    answer,_ = load_image('./eval-data/mini_evals/'+img_no+'_clean.png')
    cv2.imshow('answer',answer); cv2.waitKey(0)
    expected = answer * mask

    max_err_img = cv2.imread('./eval-data/mini_evals/'+img_no+'_clean.png')

    max_err_img = cv2.bitwise_not(max_err_img)
    #cv2.imshow('max error img',max_err_img); cv2.waitKey(0)
    #print(np.sum(max_err_img))

    max_err_img = (max_err_img.astype(np.float32) / 255) * mask
    #cv2.imshow('masked max error img',max_err_img); cv2.waitKey(0)
    #print(np.sum(max_err_img))
    #print(answer.shape)
    #max_err_img = cv2.bitwise_not(answer.astype()).astype(np.float32) * mask
    actual = completed * mask

    #cv2.imshow('expected',expected); cv2.waitKey(0)
    #cv2.imshow('actual',actual); cv2.waitKey(0)
    #cv2.imshow('max error img',max_err_img); cv2.waitKey(0)

    result_mse = mse(expected,actual)
    max_mse = mse(expected,max_err_img)
    print('mse =', result_mse)
    print('max mse =', max_mse)
    print('similarity = {:3f}%'.format((max_mse - result_mse) / max_mse * 100))
    print('error = {:3f}%'.format(100 - (max_mse - result_mse) / max_mse * 100) )

    from skimage.measure import compare_ssim
    masked_ssim = compare_ssim(expected[:,:,0],actual[:,:,0]) # inputs must be 2D array!
    print('masked ssim = {}'.format(masked_ssim))
    full_ssim = compare_ssim(answer[:,:,0],completed[:,:,0]) # inputs must be 2D array!
    print('full ssim = {}'.format(full_ssim))

if __name__ == '__main__':
    unittest.main()
    #main()
