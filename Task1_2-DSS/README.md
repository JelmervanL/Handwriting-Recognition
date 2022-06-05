# Handwriting Recognition 2022 | Task 1 & 2: Dead Sea Scrolls

## Group 05
- Joost Franssen (s3210103)
- Eden Heijnekamp (s3749185)
- Jelmer van Lune (s3128806)

## Overview of folder and files
### Folders
- `deprecated` contains deprecated data augmentation code which is no longer used
- `binary_scrolls` contains binarized Dead Sea Scrolls to test our system
- `monkbrill-jpg` contains images of the Hebrew characters used for training our classifier
- `models` contains the trained CNN classifier model

### Files
- `monkbrill_data_analysis.ipynb` contains an exploratory data analysis of the monkbrill training data in `monkbrill-jpg/`
- `train_classifier.py` contains code to train our CNN classifier using the monkbrill training data in `monkbrill-jpg/`
- `line_segmentation.py`, `word_segmentation.py` and `char_segmentation.py` contain the functions for line, word and character segmentation respectively.
- `classify.py` contains the functions to classify the segmented characters per line and write it to a txt file
- `main.py` is the main executable and can be used to run the whole pipeline. This is explained in more detail below.

## Short explanation of methods
First for every binarized input Dead Sea scroll image we segment the text lines. This is done by finding the horizontal projection profile of the image and detecting its peaks. Then the A* path planning algorith is utilized to find paths for segmentation lines. After this word segmentation on the segmented lines is performed. Calculating the horizontal projections of black pixels is used to segment words. If a white pixel area is longer than a specific threshold, we use it as the starting and stopping point for various words. A contour detection approach is used to segment the characters. If a bounding box of contours was larger than a particular constant number, it was likely to include many characters that needed to be separated. The separation for these characters is accomplished by looping the bounding box image while eroding the characters, resulting in the discovery of new boundingboxes for the characters. If that didn't work, we made a custom divider that uses a peak finder to split a word. Finally, these segmented character images are classified using a trained LeNet CNN and then written to a txt file.


## Running the code

First of all, make sure Python 3 is installed. 

### Setup virtualenv with required packages

It would be best to create a virtual environment to install the required packages in (instead of installing globally). 

To create the `virtualenv` and install the required packages on Linux/Unix:

```bash
pip3 install virtualenv
python3 -m venv ./venv
source ./venv/bin/activate
pip3 install -r requirements.txt
```

Or on Windows in cmd:

```bash
pip3 install virtualenv
python3 -m venv ./venv
.\venv\Scripts\activate.bat
pip3 install -r requirements.txt
```

### Running the full pipeline

The full pipeline can be ran using:
```bash
python3 main.py <path to input directory with jpg images>
```

By default the results of each input image will be saved in a txt file in a newly created `results` folder.

**Note**: The line segmentation for an input image can take quite long, because the A* algorithm is used which is computationally expensive.

Use:
```bash
python3 main.py --help
```

to see the possible command line arguments. This outputs:
```
python3 main.py --help
usage: main.py [-h] [-o OUTPUT_DIR] input_dir

positional arguments:
  input_dir             Path to input that contains the binarized images of scrolls

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        output directory where the txt file for each image will be saved (default: ./results/)
```

### Training the classifier

To train the CNN classifier use:
```bash
python3 train_classifier.py
```

The trained model will be saved in `models/trained_LeNet_model.h5`.


**Note**: This is not necessary before running the full pipeline because a trained model is already saved in the `models` folder.


