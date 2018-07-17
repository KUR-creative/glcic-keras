import os, sys, secrets, cv2, yaml
import numpy as np
from tqdm import tqdm
from itertools import product

from keras.layers import Input, Add, Multiply, merge
from keras.models import Model
from keras.utils import plot_model
from skimage.measure import compare_ssim

from fp import pipe, cmap, flip, unzip
from layers import completion_net, discrimination_net
from tester_ui import tester_ui
import utils


utils.help_option(
'''
evaluator: 
  evaluate all complnets in 'complnet_dir' 
  using (origin,answer,mask) in 'dataset_dir'.
  
  save mse ratio similarity/error, masked/full ssim as yml.
  save mean of scores and list of all scores.

[synopsis]
python evaluator.py complnet_dir dataset_dir img_dir

ex)
python evaluator.py olds/192x_200e/ eval-data/mini_evals ./output/small30_1000results/
'''
)

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
    #cv2.imshow('li',origin);cv2.waitKey(0)
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
    if max_mse == 0:
        return np.float_(1.0)
    else:
        return (max_mse - result_mse) / max_mse

cnet_path = None
index = 0
def scores(compl_model, origin, mean_mask, not_mask, answer, debug=False):
    completed = completion(compl_model, origin, mean_mask, not_mask)

    global cnet_path, index
    #cv2.imshow('wtf', completed); cv2.waitKey(0)
    p = os.path.join(sys.argv[3], # directory to save completed images. 
                     (cnet_path + str(index)).replace(os.sep,'_')  + '.png')
    #print(p,flush=True)
    cv2.imwrite(p, inverse_normalized(completed)) 
    index += 1

    answer_uint8 = inverse_normalized(answer)
    max_err_img = np.bitwise_not(answer_uint8)

    mask = np.logical_not(not_mask).astype(np.float32)

    masked_completed = completed * mask
    masked_answer = answer * mask
    masked_max_err_img = normalized(max_err_img) * mask

    similarity = mse_ratio_similarity(masked_completed, masked_answer, 
                                      masked_max_err_img)
    error = 1 - similarity
    masked_ssim = compare_ssim(masked_answer[:,:,0],masked_completed[:,:,0]) # inputs must be 2D array!
    full_ssim = compare_ssim(answer[:,:,0],completed[:,:,0]) # inputs must be 2D array!
    #-------------------------------------------------------------
    if debug:
        cv2.imshow('origin',origin); 
        cv2.imshow('mean_mask',mean_mask); 
        cv2.imshow('not_mask',not_mask); 
        cv2.imshow('completed',completed); 
        cv2.imshow('answer',answer); 
        cv2.imshow('max error img',max_err_img); cv2.waitKey(0)
    #-------------------------------------------------------------
    return similarity, error, masked_ssim, full_ssim

def path_tuples(answer_paths, mask_paths):
    '''yield (origin, answer, mask)paths'''
    #for pair in product(answer_paths, mask_paths):
    for answer_paths in answer_paths:
        yield answer_paths, answer_paths, secrets.choice(mask_paths)
        #yield pair[0], pair[0], pair[1]

def path_tup2img_tup(origin_path, answer_path, mask_path):
    origin = utils.slice1channel(load_image(origin_path))
    answer = np.copy(origin)

    mean_pixel_value = np.mean(origin)
    hwc = origin.shape

    mean_mask, not_mask = load_mask_pair(mask_path, mean_pixel_value)
    mean_mask = utils.hw2hwc(mean_mask)
    not_mask = utils.hw2hwc(not_mask)

    mean_mask = adjusted_image(mean_mask,hwc)
    not_mask = adjusted_image(not_mask,hwc,1.0)
    return origin, mean_mask, not_mask, answer

def save_result(complnet_path,dataset_path):
    #-------------------------------------------------------------
    compl_model = load_compl_model(complnet_path, (None,None,1))
    global cnet_path, index
    cnet_path = complnet_path
    index = 0
    #-------------------------------------------------------------
    paths = list(utils.file_paths(dataset_path))
    mask_paths = list(filter(lambda s: 'mask' in s, paths))
    #answer_paths = list(filter(lambda s: 'clean' in s, paths))
    answer_paths = list(filter(lambda s: not('mask' in s), paths))
    #print(len(answer_paths),answer_paths,flush=True)

    test_infos, similarities, errors, masked_ssims, full_ssims = [],[],[],[],[]
    for path_tup in path_tuples(answer_paths, mask_paths):
        similarity, error, masked_ssim, full_ssim\
            = scores(compl_model, *path_tup2img_tup(*path_tup))
        test_infos.append(path_tup)
        similarities.append(np.asscalar(similarity))
        errors.append(np.asscalar(error))
        masked_ssims.append(np.asscalar(masked_ssim))
        full_ssims.append(np.asscalar(full_ssim))
        #-------------------------------------------------------------
    result = {'name' : (complnet_path.replace(os.sep,'_') 
                        + '+' + 
                        dataset_path.replace(os.sep,'_')),
              'cnet_path' : complnet_path,
              'dataset_path' : dataset_path,

              'mse ratio similarity mean' : sum(similarities) / len(similarities),
              'mse ratio error mean' : sum(errors) / len(similarities),
              'masked ssim mean' : sum(masked_ssims) / len(similarities),
              'full ssim mean' : sum(full_ssims) / len(similarities),

              'origin,answer,mask paths' : test_infos,
              'similarities' : similarities,
              'errors' : errors,
              'masked_ssims' : masked_ssims,
              'full_ssims' : full_ssims}

    with open(result['name']+'.yml','w') as f:
        f.write(yaml.dump(result))

    print('{}'.format(result['name']),end='|') 
    print('mse ratio similarity mean = {:f} ({:f}%)'\
            .format(result['mse ratio similarity mean'], 
                    result['mse ratio similarity mean']*100),end='|')
    print('mse ratio error mean = {:f} ({:f}%)'\
            .format(result['mse ratio error mean'],
                    result['mse ratio error mean']*100),end='|')
    print('masked ssim mean = {:f}'.format(result['masked ssim mean']),end='|')
    print('full ssim mean = {:f}'.format(result['full ssim mean']),flush=True)


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

import re
def human_sorted(iterable):
    ''' Sorts the given iterable in the way that is expected. '''
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(iterable, key = alphanum_key)
        
def main(complnet_dir,dataset_dir):
    complnet_paths = utils.file_paths(complnet_dir)
    complnet_paths = list(human_sorted(complnet_paths))
    #complnet_paths = list(reversed(human_sorted(complnet_paths)))
    for complnet_path in tqdm(complnet_paths):
        save_result(complnet_path, dataset_dir)

if __name__ == '__main__':
    if len(sys.argv) != 3+1:
        print(' [usage]\npython evaluator.py complnet_dir dataset_dir img_dir')
    else:
        main(sys.argv[1],sys.argv[2])
    #main('olds/192x_200e/','eval-data/mini_evals')
