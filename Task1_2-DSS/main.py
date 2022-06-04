import sys, os
import argparse
from argparse import ArgumentDefaultsHelpFormatter
from line_segmentation import *
from word_segmentation import *
from char_segmentation import *
from classify import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-o", "--output_dir", type=str, default="./results/", help="output directory where the txt file for each image will be saved")
    parser.add_argument("input_dir", help="Path to input that contains the binarized images of scrolls")
    args = parser.parse_args()


    img_dir = args.input_dir
    output_dir = args.output_dir

    # create results directory for output
    if not os.path.exists(output_dir):
	    os.makedirs(output_dir)

    print ("")
    print("Input image directory:", img_dir)
    print("Results will be saved in:", output_dir)
    print ("")
    print("------------------------------")

    for input_img in os.listdir(img_dir):

        image = cv2.imread(os.path.join(img_dir, input_img), cv2.IMREAD_UNCHANGED)
        image_name = input_img.split('.')[0]

        # Start segmenting text lines
        print("Start segmenting lines " + input_img)
        line_images = line_segmentation(image)
        print("Done segmenting lines " + input_img)
        print("   #####   ")

        # Segment words in each line
        print("Start segmenting words " + input_img)
        words_per_line = get_words_per_line(line_images)
        print("Finished segmenting words " + input_img)
        print("   #####   ")

        # Segment characters,classify them and write to output file
        print("Start segmenting characters and classification of " + input_img)
        write_to_txt(words_per_line, os.path.join(output_dir, image_name + "_characters.txt"))
        print("Finished classifying " + input_img)
        print("")

        output_file = os.path.join(output_dir, image_name + "_characters.txt")
        print("Output is in:", output_file)

        print("------------------------------")

    print("Finished for all images in input directory")
    print("All results are in", output_dir)