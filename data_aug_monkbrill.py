import os 
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage import transform 
from skimage.util import random_noise
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Function for random cutout in picure
def get_random_eraser(p=1, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        if input_img.ndim == 3:
            img_h, img_w, img_c = input_img.shape
        elif input_img.ndim == 2:
            img_h, img_w = input_img.shape

        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            if input_img.ndim == 3:
                c = np.random.uniform(v_l, v_h, (h, w, img_c))
            if input_img.ndim == 2:
                c = np.random.uniform(v_l, v_h, (h, w))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w] = c

        return input_img

    return eraser

# function for elastic transform 
def elastic_transform(image, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    h, w = image.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    dx = gaussian_filter((random_state.rand(h,w) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(h,w) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    
    if len(image.shape) > 2:
        c = image.shape[2]
        distored_image = [map_coordinates(image[:,:,i], indices, order=1, mode='reflect') for i in range(c)]
        distored_image = np.concatenate(distored_image, axis=1)
    else:
        distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    
    return distored_image.reshape(image.shape)

### Augmentation functions ###
def keep_original(img):
    return img

def erosion(img):
    kernel_size = random.randint(2, 4)
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    return img

def dilation(img):
    kernel_size = random.randint(2, 4)
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    return img

def rotate(img):
    rows = img.shape[0]
    cols = img.shape[1]

    img_center = (cols / 2, rows / 2)
    M = cv2.getRotationMatrix2D(img_center, random.randint(-15, 15), 1)
    rotated_image = cv2.warpAffine(img, M, (cols, rows),
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(255))

    return rotated_image

def cutout(img):
    eraser = get_random_eraser(p=1, s_l=0.05, s_h=0.3, r_1=0.25, r_2=1/0.25, v_h=255, v_l=255, pixel_level=False)
    img = eraser(img)
    return img

def shear(img):
    afine_tf = transform.AffineTransform(shear=random.uniform(-0.3, 0.3))
    img = transform.warp(img, inverse_map=afine_tf, cval=1, mode='constant')
    img = (255*img).astype(np.uint8)
    return img

def salt_pepper_noise(img):
    noise_img = random_noise(img, mode='s&p', amount = random.uniform(0.05, 0.3), salt_vs_pepper = random.uniform(0.4, 0.6))
    noise_img = (255*noise_img).astype(np.uint8)
    return noise_img

def elastic(img):
    transformed_image=elastic_transform(img,(random.randint(-30, 30)), (random.randint(3, 7)))
    return transformed_image

if __name__ == '__main__':

    # Parse command line arguments
    parser = ArgumentParser(description = '', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_dir", default = "monkbrill-jpg/", help="Path to input directory that contains the image that need to be augmented")
    parser.add_argument("-o", "--save_dir", default="monkbrill-jpg-augmented/", help="Name of output directory where the augmented images are saved")
    parser.add_argument("-n", "--num_files", default=20000, type = int, help="Number of augmented files to be generated")
    args = vars(parser.parse_args())

    data_path = args["input_dir"]
    save_path = args["save_dir"]
    num_files_desired = args["num_files"]

    print("Number of images that will be generated:", num_files_desired)
    print("Images will be saved in:", save_path)

    # dictionary of the transformations we defined earlier
    available_transformations = {
    'original': keep_original,
    'rotation': rotate,
    'dilation': dilation,
    'erosion': erosion,
    'cutout': cutout,
    'shear': shear,
    'noise': salt_pepper_noise,
    'elastic': elastic
    }

    # create save directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Create new augmented images and save in folder
    # first a random character is chosen with even probability
    # Then random image of that character is chosen
    # Then one of the augmentation functions is applied and the image is saved

    num_generated_files = 0
    while num_generated_files < num_files_desired:

        random_char = random.choice(os.listdir(data_path))
        random_folder = os.path.join(data_path, random_char)
        random_file = random.choice(os.listdir(random_folder))

        image_to_transform = cv2.imread(os.path.join(random_folder, random_file), cv2.IMREAD_GRAYSCALE)
        
        key = random.choice(list(available_transformations))
        transformed_image = available_transformations[key](image_to_transform)

        char_save = os.path.join(save_path, random_char)
        if not os.path.exists(char_save):
            os.makedirs(char_save)
        
        cv2.imwrite(os.path.join(char_save, random_char +'_aug_' + str(num_generated_files) + "_" + key + '.jpg'), transformed_image)
        
        num_generated_files += 1
    
    print("Done")