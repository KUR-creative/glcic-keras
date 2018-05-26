import numpy as np
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

