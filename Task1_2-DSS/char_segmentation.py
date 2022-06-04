#Packages that need to be imported
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.signal import find_peaks
from heapq import *
from PIL import Image
from line_segmentation import *
from word_segmentation import *

#Find the boundingboxes of the letters
def boundingboxes_letters(image, image_dilated):
    boxcoordinates = []
    words_image = []
    letters = []

    copy = image.copy()
    thresholded = thresholding(image_dilated)

    cnts = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    for c in cnts:
        topleftx, toplefty, width, height = cv2.boundingRect(c)
        next_letter = [topleftx, toplefty, width, height]
        letters.append(next_letter)

    #Sort the letters based on the x values
    letters = sorted(letters, reverse=False)


    for letter in letters:
        topleftx, toplefty, width, height = letter

        #Reject contours that are too big or small to be likely letters, from average data taken
        #Normal minima are height = 27, width = 17
        if height<40 or width<30:
            continue

        #Normal maxima are height = 140, width = 80
        if height>300 or width>500:
            continue

        cv2.rectangle(copy, (topleftx, toplefty), (topleftx + width, toplefty + height), (0,0,255), 2)  
        boxcoordinates.append((topleftx, toplefty, width, height))

        character = image[toplefty:toplefty+height, topleftx:topleftx+width]
        words_image.append(character)
    
    return words_image, copy, boxcoordinates

#Find the middle part of a letter, using a peak finder on where the most vertical projections are
def find_middle_part_letter(image):
    hpp_image = thresholding(image)
    hpp_image = horizontal_projections(hpp_image)
    minimum_peak = int((np.max(hpp_image)-np.min(hpp_image))/3)
    peaks, _ = find_peaks(hpp_image, height=minimum_peak , distance=100) 
    if len(peaks) != 0:
        toplefty = max(peaks[0]-20, 0)
        height = min(40, image.shape[0]-toplefty)
        new_image = image[toplefty:toplefty+height, :]
    else:
        new_image = image
    return new_image

#Calculate the horizontal projections and find the peaks of white areas
def not_bb_divider_peaks(image):
    vpp = vertical_projections(image)
    minimum_peak = int((np.max(vpp)-np.min(vpp))/3)
    peaks, _ = find_peaks(vpp, height=minimum_peak , distance=60)
    new_peaks = []
    for peak in peaks:
        #peaks lower than x value of 30 or 30 from end are not really dividers 
        if peak < 27:
            continue
        elif peak > (image.shape[1] - 27):
            continue
        else:
            new_peaks.append(peak)
        
    return new_peaks
    
#Divided the characters that are too large based on the peaks found in not_bb_divider_peaks()
def divided_chars(image, peaks):
    chars = []
    if len(peaks) == 0:
        first_char = image[:,:]
    else:
        first_char = image[:,0:peaks[0]]
    chars.append(first_char)
    for i in range(len(peaks)):
        if i+1 == len(peaks):
            char = image[:,peaks[i]:image.shape[1]]
        else:
            char = image[:,peaks[i]:peaks[i+1]]
        chars.append(char)
    return chars


#Handcrafted function to divided characters that have too large boundingboxes to be one letter
def not_bb_divided(image):
    character = []
    middle_part = find_middle_part_letter(image)
    peaks_divider = not_bb_divider_peaks(middle_part)
    characters = divided_chars(image, peaks_divider)
    for i in range(len(characters)):
        if characters[i].shape[0] == 0 or characters[i].shape[1] == 0:
            continue 
        else:
            character.append(characters[i])
    return character

   
    
#If parts of the word are not obtained as bounding box we add them manually
def adjust_bb_for_mistakes(word_bb_image, coordinates, word, char):
    count = 1
    if len(coordinates) != 0:
        for i in range(len(coordinates)-1):
            toprightx = coordinates[i][0]+coordinates[i][2]
            distance_between_bb = coordinates[i+1][0] - toprightx
            
            #If the boundingboxes do not connect, we create a new boundingbox in the middle
            if distance_between_bb > 25:
                character = word[:, toprightx:coordinates[i+1][0]]
                char.insert(i+count, character)
                count+=1
            
        #Check if at the end of the bb range there is still something left
        toprightx = coordinates[len(coordinates)-1][0] + coordinates[len(coordinates)-1][2]
        if (toprightx) < word_bb_image.shape[1] - 15:
            char.pop()
            character = word[:,coordinates[len(coordinates)-1][0]:]
            char.append(character)

        #Check if the first boundingbox starts at the beginning of the word
        if coordinates[0][0] > 0 and len(coordinates) > 1:
            new_width = coordinates[0][0] + coordinates[0][2]
            char.pop(0)
            character = word[:,0:new_width]
            char.insert(0, character)      
    return char

#Function to erode images such that we hope to find more boundingboxes in the too large images to be one character
def erosion_char_finder(double_char):
    characters = []
    erosion_level = 2
    kernel = np.ones((erosion_level,erosion_level), np.uint8)
    double_char_dilate = cv2.dilate(double_char, kernel, iterations=1)
    chars, test_image, coordinates = boundingboxes_letters(double_char, double_char_dilate)
    
    while len(chars) == 1:
        kernel = np.ones((erosion_level,erosion_level), np.uint8)
        double_char_dilate = cv2.dilate(double_char, kernel, iterations=1)
        chars, test_image, coordinates = boundingboxes_letters(double_char, double_char_dilate)
        erosion_level += 1
        
    for x in range(len(chars)):
        #Too avoid that if we have a bb above another bb that it creates a new unwanted character
        if (x + 1) < len(chars) and coordinates[x+1][0] > coordinates[x][0] and coordinates[x+1][2] < (coordinates[x][2] - coordinates[x+1][0]):
            characters.append(double_char)
            break
        else:
            topleftx = max((coordinates[x][0] - 2), 0)

            #Often if we have two conjoint letters we have a too small bounding box for the last letter
            #Therefore we take the full bb of everything from the last letter onwards. 
            if x == (len(chars) -1):
                width = double_char.shape[1]-coordinates[x][0]
            else:
                width = min((coordinates[x][2] + 2), (double_char.shape[1]-coordinates[x][0]))
            new_char = double_char[:, topleftx:topleftx+width]
            characters.append(new_char)
    return characters

def find_characters_in_word(word):
    #character finder in total:
    word_copy = word.copy()

    #the first time we have a word we dilate such that we can find letters that are already seperated form others
    kernel = np.ones((3,3), np.uint8)
    word_copy_dilate = cv2.erode(word_copy, kernel, iterations=1)

    char, word_bb_image, coordinates = boundingboxes_letters(word_copy, word_copy_dilate)     
    
    #Adjust the charachters bb in a word for missing parts
    char = adjust_bb_for_mistakes(word_bb_image, coordinates, word, char)
    
    characters = []

    #If we do not find boundingboxes within the word, the word itself is a letter, or it does not contain clear bbs
    if len(char) == 0:
        if word_copy.shape[1] > 100:
            characters_handcrafted_div = not_bb_divided(word_copy)
            for l in range(len(characters_handcrafted_div)):
                characters.append(characters_handcrafted_div[l])
        else:
            characters.append(word)
    else:
        for i in range(len(char)):
            if char[i].shape[1] > 85:
                new_char_bb = erosion_char_finder(char[i])
                
                
                if len(new_char_bb) == 0:
                    #Still searching for the best parameter for this
                    if char[i].shape[1] > 90:
                        #this is a handcrafted divider
                        characters_handcrafted_div = not_bb_divided(char[i])
                        for u in range(len(characters_handcrafted_div)):
                            characters.append(characters_handcrafted_div[u])
                    else:
                        characters.append(char[i])
                else:
                    for x in range(len(new_char_bb)):
                        if new_char_bb[x].shape[1] > 90:
                            new_new_char_bb = erosion_char_finder(new_char_bb[x])
                            if len(new_new_char_bb) == 0:
                                if new_char_bb[x].shape[1] > 86:
                                    characters_handcrafted_div = not_bb_divided(new_char_bb[x])
                                    for u in range(len(characters_handcrafted_div)):
                                        characters.append(characters_handcrafted_div[u])
                                else:
                                    characters.append(new_char_bb[x])
                            else:
                                if len(new_new_char_bb) == 1:
                                    if new_new_char_bb[0].shape[1] > 85:
                                        characters_handcrafted_div = not_bb_divided(new_char_bb[0])
                                        for u in range(len(characters_handcrafted_div)):
                                            characters.append(characters_handcrafted_div[u])
                                else:
                                    for y in range(len(new_new_char_bb)):
                                        characters.append(new_new_char_bb[y])
                        else:
                            characters.append(new_char_bb[x])
            else:
                characters.append(char[i])
    return characters

#################
### Main function to find the all the characters in a line
################
def characters_per_line(words):
    char_per_word = []
    for i in range(len(words)):
        characters = find_characters_in_word(words[i])
        white_parts_char_removed = []
        
        #Remove unnecessary white parts at the top and bottom of characters
        for j in range(len(characters)):
            char_bot = remove_bottom_image(characters[j])
            char_top = remove_top_image(char_bot)
            white_parts_char_removed.append(char_top)
        if len(characters) != 0:
            char_per_word.append(white_parts_char_removed)
        else:
            char_per_word.append(characters)
    return char_per_word   