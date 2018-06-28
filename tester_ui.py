import cv2
import numpy as np
drawing = False # true if mouse is pressed
mode = False # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
# mouse callback function
def draw_mask(event,x,y,flags,params):
    global ix,iy,drawing,mode
    mask,pen_size = params
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(mask,(ix,iy),(x,y), (0,0,0), -1)
            else:
                cv2.circle(mask,(x,y),pen_size, (0,0,0), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(mask,(ix,iy),(x,y), (0,0,0), -1)
        else:
            cv2.circle(mask,(x,y),pen_size, (0,0,0), -1)

def tester_ui(bg): # bg is standardized image(img / 255)
    global mode
    not_mask = np.ones(bg.shape, np.float32)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_mask, (not_mask,3))

    screen = np.empty(bg.shape)
    while(1):
        screen = bg.astype(np.float32) * not_mask 
        color_mask = np.logical_not(not_mask).astype(np.float32)
        color_mask *= (0,0,1) # red mask
        screen += color_mask
        cv2.imshow('image',screen)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == 13: #ESC = 27, CarriageReturn(enter) = 13
            break

    mask = np.logical_not(not_mask).astype(np.float32)
    return mask

if __name__ == '__main__':
    bg_img = cv2.imread('./data/test_images/images 2.jpg') / 255
    mask = tester_ui(bg_img) 
    cv2.imshow('mask',mask); cv2.waitKey(0)
