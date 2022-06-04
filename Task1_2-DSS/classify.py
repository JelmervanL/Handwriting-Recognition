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
from keras.models import load_model
from char_segmentation import characters_per_line

#Load in the trained model
model = load_model('models/trained_LeNet_model.h5') 

#Get the predictions per character in a word
def make_predictions(word):
    char_resized_final = []
    for i in range(len(word)):
        char_resized = cv2.resize(word[i], (38, 48))
        char_resized_final.append(char_resized)
    char_array = np.array(char_resized_final) 
    char_array = char_array.reshape(char_array.shape[0], 48, 38, 1)
    predictions = model.predict(char_array)
    classes = np.argmax(predictions, axis = 1)
    return classes

#Create a text file in which the answers are printed
def write_to_txt(words_per_line, file_name):
    f = open(file_name,"w", encoding='utf-8')
    for i in range(len(words_per_line)):
        line_characters = characters_per_line(words_per_line[i])
        for j in range(len(line_characters)-1,-1,-1):
            prediction_word = make_predictions(line_characters[j])
            for x in range(len(prediction_word)-1, -1, -1):
                f.write(find_hewbrew_character(prediction_word[x]))
            f.write(" ")
        f.write("\n")
    f.close()

#Find the hebrew character corresponding to our number output
def find_hewbrew_character(number):
    if number == 0: #Alef
        char = "א"
    if number == 1: #Ayin
        char = "ע"
    if number == 2: #Bet
        char = "ב"
    if number == 3: #Dalet
        char = "ד"
    if number == 4: #Gimel
        char = "ג"
    if number == 5: #He
        char = "ה"
    if number == 6: #Het
        char = "ח"
    if number == 7: #Kaf
        char = "כ"
    if number == 8: #Kaf-final
        char = "ך" 
    if number == 9: #Lamed
        char = "ל"
    if number == 10: #Mem
        char = "ם"
    if number == 11: #Mem-medial
        char = "מ"
    if number == 12: #Nun-final
        char = "ן"
    if number == 13: #Nun-medial
        char = "נ"
    if number == 14: #Pe
        char = "פ"
    if number == 15: #Pe-final
        char = "ף"
    if number == 16: #Qof
        char = "ק"
    if number == 17: #Resh
        char = "ר"
    if number == 18: #Samekh
        char = "ס"
    if number == 19: #Shin
        char = "ש"
    if number == 20: #Taw
        char = "ת"
    if number == 21: #Tet
        char = "ט"
    if number == 22: #Tsadi-final
        char = "ץ"
    if number == 23: #Tsadi-medial
        char = "צ"
    if number == 24: #waw
        char = "ו"
    if number == 25: #Yod
        char = "י"
    if number == 26: #Zayin
        char = "ז"
    return char