from line_segmentation import *
from word_segmentation import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

if __name__ == '__main__':
    ## TODO: need to loop over all images in directory

    # Load image 
    scroll = "binary/P168-Fg016-R-C01-R01-binarized.jpg"
    image = cv2.imread(scroll, cv2.IMREAD_UNCHANGED)

    # Start segmenting image lines
    print("Start segmenting lines")
    line_images = segment_lines(image)
    print("Done segmenting lines")

    # Segment words in each line
    print("Start segmenting words on each line")
    words_per_line = segment_words(line_images)
    print("Done segmenting words on each line")

