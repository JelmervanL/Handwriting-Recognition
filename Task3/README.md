# Handwriting Recognition 2022 | Task 3: IAM data

## Group 05
- Joost Franssen (s3210103)
- Eden Heijnekamp (s3749185)
- Jelmer van Lune (s3128806)

## Setup virtualenv with required packages

First of all, make sure Python 3 is installed. It would be best to create a virtual environment to install the required packages in (instead of installing globally). 

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

## Instructions to run code

The project already contains the IAM data used for training and validating our model. This data is in the  `IAM-data` folder.

The model used for this task is the [TrOCR model](https://arxiv.org/abs/2109.10282). The training of this model is performed using the [Huggingface Transformer Library](https://huggingface.co/docs/transformers/model_doc/trocr). We finetuned a pre-trained model (pre-trained on printed text and synthesized handwritten text) on the provided IAM dataset.
A jupyter notebook is used for training, and the model was trained in Google Colab. The trainig code is based on this [tutorial](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/TrOCR).

To train the model open the `task3_iam_training.ipynb` jupyter notebook and run all cells consecutively. We also already trained the model, which is saved in the `models` folder. This model reached a CER of 6.1 on the test data.

This trained model can be invoked on new data. To do this use:

```bash
python3 task3_iam_test.py --input_dir <path to directory containing the images> 
```

For each image in the input directory a txt file containing the predicited characters is saved in the now created `results` folder.

An example: 

```bash
python3 task3_iam_test.py --input_dir IAM-data/test/ 
```

Use:

```bash
python3 task3_iam_test.py --help
```

to see the possible commands. This outputs:

```
python3 task3_iam_test.py --help
usage: task3_iam_test.py [-h] [-i INPUT_DIR] [-o OUTPUT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input_dir INPUT_DIR
                        Path to input that contains the line images (default: None)
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        output directory where the txt file for each image will be saved (default: results/)
```
