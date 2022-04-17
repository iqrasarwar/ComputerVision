#!/usr/bin/env python
# coding: utf-8

# In[22]:


import cv2
import math
import numpy as np


# In[23]:


img = cv2.imread('C:/Users/BUTT SYSTEMS/OneDrive/Desktop/book.png')
result = np.zeros(shape = (img.shape[0],img.shape[1],3))


# In[24]:


blue_channel = img[:,:,0]
green_channel = img[:,:,1]
red_channel = img[:,:,2]


# In[78]:


def convolution(mask,img):
    convoluted = np.zeros(shape = (img.shape[0],img.shape[1]))
    r = img.shape[0]
    c = img.shape[1]
    for i in range(r):
        for j in range(c):
            # current stores the currently active portion of img being multiplied with mask
            current = np.zeros(shape = (mask.shape[0],mask.shape[1]))
            r1 = mask.shape[0]
            r2 = mask.shape[1]
            # finding the center location of the current active mask
            a = math.floor(r1/2)
            b = math.floor(r2/2)
            current[a][b] = img[i][j]
            x = 0
            y = 0
            for k in range(i-a,i+a):
                y=0
                for l in range(j-b,j+b):
                    if(k>=0 and l>=0 and k<r and l<c):
                        current[x][y] = img[k][l] 
                    y= y+1
                x=x+1
            sum = 0
            for m in range (x):
                n=0
                for n in range(y):
                    sum += current[m][n]*mask[m][n]
            convoluted[i][j] = sum
    return convoluted


# In[79]:


def performConvolution(red_channel,green_channel,blue_channel,kernel,result):
    flip(kernel)
    result[:,:,0] = convolution(kernel,blue_channel)
    result[:,:,1] = convolution(kernel,green_channel)
    result[:,:,2] = convolution(kernel,red_channel)
    return result


# In[80]:


def flip(kernel):
    kernel[[0, -1]] = kernel[[-1, 0]]
    kernel[:,[0, -1]] = kernel[:,[-1, 0]]
    return kernel


# In[81]:


kernel = 1/9*(np.array([[1,1,1],
                         [1,1,1],
                         [1,1,1]]))
result = performConvolution(red_channel,green_channel,blue_channel,kernel,result)
cv2.imwrite('C:/Users/BUTT SYSTEMS/OneDrive/Desktop/conBook3.png',result)
img = cv2.imread('C:/Users/BUTT SYSTEMS/OneDrive/Desktop/conBook3.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #matplotlib uses RGB
plt.imshow(img)


# In[82]:


kernel = 1/25*(np.array([[1,1,1,1,1],
                         [1,1,1,1,1],
                         [1,1,1,1,1],
                         [1,1,1,1,1],
                         [1,1,1,1,1]]))
result = performConvolution(red_channel,green_channel,blue_channel,kernel,result)
cv2.imwrite('C:/Users/BUTT SYSTEMS/OneDrive/Desktop/conBook5.png',result)
img = cv2.imread('C:/Users/BUTT SYSTEMS/OneDrive/Desktop/conBook5.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #matplotlib uses RGB
plt.imshow(img)


# In[83]:


kernel = 1/49*(np.array([[1,1,1,1,1,1,1],
                         [1,1,1,1,1,1,1],
                         [1,1,1,1,1,1,1],
                         [1,1,1,1,1,1,1],
                         [1,1,1,1,1,1,1],
                         [1,1,1,1,1,1,1],
                         [1,1,1,1,1,1,1]]))
result = performConvolution(red_channel,green_channel,blue_channel,kernel,result)
cv2.imwrite('C:/Users/BUTT SYSTEMS/OneDrive/Desktop/conBook7.png',result)
img = cv2.imread('C:/Users/BUTT SYSTEMS/OneDrive/Desktop/conBook7.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #matplotlib uses RGB
plt.imshow(img)


# In[84]:


kernel = (np.array([[0,-1,0],
                         [-1,5,-1],
                         [0,-1,0]]))
result = performConvolution(red_channel,green_channel,blue_channel,kernel,result)
cv2.imwrite('C:/Users/BUTT SYSTEMS/OneDrive/Desktop/sharp.png',result)
img = cv2.imread('C:/Users/BUTT SYSTEMS/OneDrive/Desktop/sharp.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #matplotlib uses RGB
plt.imshow(img)


# In[ ]:




