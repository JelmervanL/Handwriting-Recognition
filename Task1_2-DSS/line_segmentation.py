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

# Declare constants
RESCALE_FACTOR = 3500

#Calculate the reshaping height
def get_rescale_height(image):
    rescaled_height = image.shape[0]/RESCALE_FACTOR

    return rescaled_height

#Reshape the height of each image to a specific value such that the line segmentation works good for every image
def reshape_image(image):
    image = cv2.resize(image, (image.shape[1], RESCALE_FACTOR))

    return image

#Function to threshold the image
def thresholding(image):
    _ ,thresh = cv2.threshold(image,80,255,cv2.THRESH_BINARY_INV)
    return thresh

#Function to dilated an image
def dilated_image(image):
    thresh_img = thresholding(image)
    kernel = np.ones((4,35), np.uint8)
    dilated = cv2.dilate(thresh_img, kernel, iterations = 1) 
    return dilated

#Function to ivert a image
def invert_image(image):
    invert = cv2.bitwise_not(image) 
    return invert

#Calculate the horizontal projections
def horizontal_projections(image):
    return np.sum(image, axis=1) 

#Peak finder for the horizontal projections
def find_peaks_image(image):
    dilated = dilated_image(image)
    hpp = horizontal_projections(dilated)
    
    #The minimum peak assures that we have a lower bound
    minimum_peak = int((np.max(hpp)-np.min(hpp))/12)
    
    #The distance assures that we have a certain distance between peaks
    peaks, _ = find_peaks(hpp, height=minimum_peak, distance=120)
    return hpp, peaks


#Use the calculated peaks to create small areas of where there must be sentences. 
def find_not_peak_regions(hpp, peaks):
    not_peaks = []
    not_peaks_index = []
    count = 0
    x = 0
    line_size = 10
    for i, hppv in enumerate(hpp):
        if x > (len(peaks) - 1):
            not_peaks.append([i, hppv])
        elif i < (peaks[x] - line_size) or i > (peaks[x] + line_size):
            not_peaks.append([i, hppv])
        else:
            count += 1
            if count == ((line_size*2) - 1):
                x += 1
                count = 0
    return not_peaks



#The hpp clusters are created such that we have areas in which the path finder can search, creating walking regions in the image
def get_hpp_walking_regions(peaks_index):
    hpp_clusters = []
    cluster = []
    for index, value in enumerate(peaks_index):
        cluster.append(value)

        if index < len(peaks_index)-1 and peaks_index[index+1] - value > 1:
            hpp_clusters.append(cluster)
            cluster = []

        #get the last cluster
        if index == len(peaks_index)-1:
            hpp_clusters.append(cluster)
            cluster = []
            
    return hpp_clusters

#Create new line images such that we can copy the segmented line unto clean images
def white_array(height, width):
    image = np.zeros([height,width],dtype=np.uint8)
    image.fill(255)
    return image


#Get the binary values of a image
def get_binary(img):
    mean = np.mean(img)
    if mean == 0.0 or mean == 1.0:
        return img

    thresh = threshold_otsu(img)
    binary = img <= thresh
    binary = binary*1
    return binary

#A star path planning algorithm 
def heuristic(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

def astar(array, start, goal):
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []

    heappush(oheap, (fscore[start], start))
    
    lower_limit = goal[0]
    upper_limit = goal[0] + lower_limit 
    
    count = 0
    while oheap:
        count += 1
        
        current = heappop(oheap)[1]
        
        #If there is not a path break after 30 times the length of the input iterations, no clear path will probably be found
        if count > goal[1]*30:
            break
            
        #The goal of the path finder can be somewhere in the range of the walking region at the end
        if current[1] == goal[1] and (current[0] > lower_limit and current[0] < upper_limit):
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j            
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue
                
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
                
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(oheap, (fscore[neighbor], neighbor))      
    return []

#Segment all the lines using the A* algorithm, and find a path. 
def create_A_star_lines(img, hpp_clusters):
    binary_image = get_binary(img)

    line_segments = []
    for i, cluster_of_interest in enumerate(hpp_clusters):
        nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:]

        #The goal of the path finder is between the middle and the lower bound of the walking region (this seemed most effective)
        path = np.array(astar(nmap, (int(nmap.shape[0]/2), 0), (int(nmap.shape[0]/2), nmap.shape[1]-1)))

        #Finding a path is impossible, then just draw a straight line from begin to end
        if path.shape[0] == 0:
            line_new = []
            size_cluster = int(nmap.shape[0]/2)
            for i in range(nmap.shape[1] - 1):
                line_new.append([size_cluster, nmap.shape[1]-i -1])
                line = []
            line_new = np.asarray(line_new)
            offset_from_top = cluster_of_interest[0]
            line_new[:,0] += offset_from_top
            line_segments.append(line_new)
        #If we have found a path we store it
        else:
            offset_from_top = cluster_of_interest[0]
            path[:,0] += offset_from_top
            line_segments.append(path)
    return line_segments

#Create the seperate line images.
def create_line_image(line_segments, image):
    line_count = len(line_segments)
    line_image = []

    #Find the upper and lower bound, for the size of the line image
    for line_index in range(line_count-1):
        upper_bound = np.min(line_segments[line_index][:, 0])
        lower_bound = np.max(line_segments[line_index+1][:, 0])
        height = lower_bound - upper_bound
        width = image.shape[1]
        white_line_image = white_array(height, width)
        offset_top = upper_bound


        #Creates white areas around the sentences, such that bits and pieces of other sentences do not get in the line image
        for i in range(white_line_image.shape[0]):
            for j in range(white_line_image.shape[1] -1):
                if (line_segments[line_index][white_line_image.shape[1] - 2 - j][0] - i > upper_bound) or (i + offset_top > line_segments[line_index+1][white_line_image.shape[1] - 2 - j][0]) :              
                    white_line_image[i][j] = 255     
                else:
                    white_line_image[i][j] = image[upper_bound + i][j]

        line_image.append(white_line_image)
    return line_image

#Rescale the height of the lines back to the normal size
def rescale_lines(line_image, rescaled_height):
    new_line_image = []
    for i in range(len(line_image)):
        height = line_image[i].shape[0]
        new_height = int(rescaled_height * height)
        new_line = cv2.resize(line_image[i], (line_image[i].shape[1], new_height))
        new_line_image.append(new_line)
    return new_line_image

#################
### Main function for the complete line segmentation steps ###
################
def line_segmentation(image):

    #Get rescale height and resize the image so line_segmenting works better
    rescaled_height = get_rescale_height(image)
    image = reshape_image(image)

    #Find the horizontal projections and the peaks of the projections
    hpp, peaks = find_peaks_image(image)

    #find all the indexes that do not contain the peak
    not_peaks = find_not_peak_regions(hpp,peaks)
    not_peaks_index = np.array(not_peaks)[:,0].astype(int)

    #Create walking regions for the A* star algorithm path finder
    hpp_clusters = get_hpp_walking_regions(not_peaks_index)

    #dilate the image such that we find a path around the dilation
    dilated = dilated_image(image)

    #invert it back for the path finder because that is build to avoid black pixels
    img = invert_image(dilated)

    #Find the paths for the segmentation
    line_segments = create_A_star_lines(img, hpp_clusters)

    #Create the segmented line images
    line_image = create_line_image(line_segments, image)
    
    rescaled_lines = rescale_lines(line_image, rescaled_height)
    
    return rescaled_lines  


