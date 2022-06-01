from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.signal import find_peaks
from heapq import *
from PIL import Image

def thresholding(image):
    _ ,thresh = cv2.threshold(image,80,255,cv2.THRESH_BINARY_INV)
    return thresh

def horizontal_projections(image):
    return np.sum(image, axis=1) 

# Remove excess white pixels first line
def remove_top_image(image):
    thresh_image = thresholding(image)
    hpp_image = horizontal_projections(thresh_image)
    for i in range(len(hpp_image)):
        if hpp_image[i] != 0:
            image_new = image[i:, :]
            break 
    return image_new

# Remove excess white pixels last line
def remove_bottom_image(image):
    thresh_image = thresholding(image)
    hpp_image = horizontal_projections(thresh_image)
    for i in range(len(hpp_image)-1, 0, -1):
        if hpp_image[i] != 0:
            image_new = image[:i, :]
            break 
    return image_new

def vertical_projections(image):
    return np.sum(image, axis=0) 

def remove_noise(image):
    return cv2.medianBlur(image, 5)

#preprocesses the lines
def pre_process_lines(lines):
    pre_processed_lines = []
    kernel = np.ones((10,10), np.uint8)
    
    for i in range(len(lines)):
        line = lines[i]
        
        #threshold line
        thresh_line = thresholding(line)
        
        #remove noise
        noise_free_line = remove_noise(thresh_line)
        
        #dilate the remaining part
        dilated_line = cv2.dilate(noise_free_line, kernel, iterations = 1)
        
        #append the line
        pre_processed_lines.append(dilated_line)

    return pre_processed_lines

def find_words_for_zero_period(image, threshold):
    words = []
    vpp_image = vertical_projections(image)
    
    #find indexes where the vpp is not zero, so there are black pixels
    result_zeros = np.where(vpp_image!=0)
    lowest = result_zeros[0][0]
    for i in range(len(result_zeros[0])-1):
        diff = result_zeros[0][i+1] - result_zeros[0][i] 
        
        #skip white periods that are not long enough
        if diff > 1 and diff < threshold:
            continue
        elif diff > 1:
            highest = result_zeros[0][i]
            new_word = [lowest, highest]
            words.append(new_word)
            lowest = result_zeros[0][i+1]
    return words 

def get_word_image(image, index_words):
    words = []
    for i in range(len(index_words)):
        #Get word
        word_new = image[:,index_words[i][0]:index_words[i][1]]
        
        #remove the white area at the top and bottom
        word_bot = remove_bottom_image(word_new)
        word_top = remove_top_image(word_bot)
        
        words.append(word_top)
    return words

def segment_words(line_images):
    first_line = line_images[0]
    last_line = line_images[len(line_images)-1]
    line_images[0] = remove_top_image(first_line)
    line_images[len(line_images)-1] = remove_bottom_image(last_line)   

    pre_processed_lines = pre_process_lines(line_images)

    words_per_line = []
    for line in range(len(line_images)):
        index_words = find_words_for_zero_period(pre_processed_lines[line], 10)
        words = get_word_image(line_images[line], index_words) 
        words_per_line.append(words)

    return words_per_line

    





