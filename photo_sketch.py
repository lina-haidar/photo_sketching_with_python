import cv2
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def initialize_kernel(size , sigma):

  """ This function initializes a gaussian kernel with the desired size and sigma. """

  #fig = plt.figure()
 # ax = fig.add_subplot(111, projection='3d')
  
  w, h = size
  
  x = np.linspace(-1,1,w)
  y = np.linspace(-1,1, h)
  x_cor, y_cor  = np.meshgrid(x, y)
  
  kernel = 1/(2*np.pi*np.power(sigma,2) ) *np.exp((- (x_cor ** 2 + y_cor ** 2) )/ (2*np.power(sigma,2)))
  
  """ Gaussion function: 1/(2 *pi*sigma^2) e^(-(x^2+y^2)/2sigma^2) """
  
 
  
  kernel = kernel/np.sum(kernel) # normalization 
  print(kernel)

 # ax.plot_surface(x_cor, y_cor, kernel)

 # plt.show()
 # print(kernel)
  return kernel

def padding(image):

  padded_image = np.pad(image , ((1,1),(1,1)) , 'constant', constant_values=(0,0) )

  return padded_image
  
def conv2d(image, ftr):
    s = ftr.shape + tuple(np.subtract(image.shape, ftr.shape) + 1)
    sub_image = np.lib.stride_tricks.as_strided(image, shape = s, strides = image.strides * 2)
    return np.einsum('ij,ijkl->kl', ftr, sub_image) 
    


def grayscale_image(image):
  # convert to grayscale 
  gray_image = np.zeros((image.shape[0],image.shape[1]),np.uint8)

  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      #gray_image[i,j] = np.clip( (image[i,j,0] + image[i,j,1] + image[i,j,2] )/3, 0, 255) # using average method
      gray_image[i,j] = np.clip(0.07 * image[i,j,0]  + 0.72 * image[i,j,1] + 0.21 * image[i,j,2], 0, 255) # using luminosity method
     
      # display the image
      
  #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  cv2.imshow('image',gray_image)
  cv2.waitKey(0)
    
  return gray_image

def blurring_image(gray_image):

  # blur the gray_image by a Gaussian function 
  blured_image = conv2d( padding(gray_image) ,  initialize_kernel((3,3) , 10) ) 
  
  # display the result
  cv2.imshow('image',blured_image/255)
  cv2.waitKey(0)

  return blured_image

def photo_sketching(image):
  # step 1
  gray_image = grayscale_image(image)
  # step 2
  blurred_image = blurring_image(gray_image)
  # step 3
  photo_sketch = np.divide(gray_image, blurred_image)

  # display the result
  cv2.imshow('image',photo_sketch )
  cv2.waitKey(0)
  
  cv2.imwrite('/home/lina/Desktop/horse1.jpeg', photo_sketch*255)


if __name__ == "__main__": 

  im_location = '/home/lina/Desktop/'

  file_name = 'horse.jpeg'

  # read the image 
  image = cv2.imread(im_location+file_name) 
  # display the image
  cv2.imshow('image',image)
  cv2.waitKey(0)

  photo_sketching(image)






