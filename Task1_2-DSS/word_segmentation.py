from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.signal import find_peaks
from heapq import *
from PIL import Image
from line_segmentation import thresholding, horizontal_projections

#Remove the top white part of a image
def remove_top_image(image):
    thresh_image = thresholding(image)
    hpp_image = horizontal_projections(thresh_image)
    for i in range(len(hpp_image)):
        if hpp_image[i] != 0:
            image_new = image[i:, :]
            break 
    return image_new

#Remove the bottom white part of a image
def remove_bottom_image(image):
    thresh_image = thresholding(image)
    hpp_image = horizontal_projections(thresh_image)
    for i in range(len(hpp_image)-1, 0, -1):
        if hpp_image[i] != 0:
            image_new = image[:i, :]
            break 
    return image_new  

#Find the vertical image projections
def vertical_projections(image):
    return np.sum(image, axis=0) 

#Remove noise from a image using the meadian blur
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

#Remove the white areas that are found in the first and last line
def remove_white_areas_first_last_line(line_image):
    first_line = line_image[0]
    last_line = line_image[len(line_image)-1]

    line_image[0] = remove_top_image(first_line)
    line_image[len(line_image)-1] = remove_bottom_image(last_line)
    return line_image


#Find the length of periods with white pixels and create seperate words if there are enough white pixels
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
              
    if lowest + 35 < result_zeros[0][len(result_zeros[0])-1]:
        highest = result_zeros[0][len(result_zeros[0])-1]
        new_word = [lowest, highest]
        words.append(new_word)
        lowest = result_zeros[0][i+1]       
    
    return words 

#Create the image for one specific word in which we remove the top and bottom white spaces
def get_word_image(image, index_words):
    words = []
    for i in range(len(index_words)):
        #Get word
        word_new = image[:,index_words[i][0]:index_words[i][1]]
        if word_new.shape[1] < 15:
            continue 
        words.append(word_new)
    return words


#Create the word image for all lines
def create_words(line_image, pre_processed_lines):
    words_per_line = []
    for line in range(len(line_image)):
        index_words = find_words_for_zero_period(pre_processed_lines[line], 10)
        words = get_word_image(line_image[line], index_words) 
        words_per_line.append(words)
    return words_per_line

#################
### Main function to create all the lines and words per line in this part
################

def get_words_per_line(line_image):

    line_image = remove_white_areas_first_last_line(line_image)

    #Pre process all the lines such that we can use them for the word and character segmentation
    pre_processed_lines = pre_process_lines(line_image)

    #Get the words that can be segmented on each line
    words_per_line = create_words(line_image, pre_processed_lines)

    return words_per_line

    





