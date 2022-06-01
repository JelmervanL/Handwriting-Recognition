from line_segmentation import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

if __name__ == '__main__':
    ## TODO: need to loop over all images in directory

    # Load image 
    scroll = "binary/P168-Fg016-R-C01-R01-binarized.jpg"
    image = cv2.imread(scroll, cv2.IMREAD_UNCHANGED)

    # Start segmenting image lines
    print("Start segmenting lines")
    # thresh_img = thresholding(image)
    # dilated = dilation(thresh_img)
    # invert = invert(dilated)

    # hpp = horizontal_projections(dilated)
    # peaks = get_peaks(hpp)

    # not_peaks = find_not_peak_regions(hpp, peaks)
    # not_peaks_index = np.array(not_peaks)[:,0].astype(int)

    # hpp_clusters = get_hpp_walking_regions(not_peaks_index)

    # binary_image = get_binary(invert)
    # line_segments = get_line_segments(hpp_clusters, binary_image)

    # line_images = get_line_images(line_segments, image)

    line_images = segment_lines(image)

    print("Done segmenting lines")

    for img in line_images:
        plt.figure(figsize=(10,10))
        plt.imshow(img, cmap="gray")
        plt.show()
