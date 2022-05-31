# Handwriting Recognition 2022 | Group 05

## Before running any pipeline

First of all, make sure python is installed. 
Then install the required packages for this project using:

```bash
pip3 install -r requirements.txt
```

## Task 1&2: Dead Sea Scrolls



## Task 3: IAM dataset

Change directory to the `Task3` folder:

```bash
cd Task3/
```

The project already contains the IAM data used for training and validating our model. This data is in the  `IAM-data` folder.

The model used for this task is the [TrOCR model](https://arxiv.org/abs/2109.10282). The training of this model is performed using the [Huggingface Transformer Library](https://huggingface.co/docs/transformers/model_doc/trocr). We finetuned a pre-trained model (pre-trained on printed text and synthesized handwritten text) on the provided IAM dataset.
For this training a jupyter notebook is used, and the model was trained in Google Colab. The trainig code is based on a [tutorial](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/TrOCR).

To train the model open the 'task3_iam_training.ipynb' jupyter notebook and run all cells consecutively. We also already trained the model, which is saved in the 'models' folder. This model reached a CER of 6.1 on the test data.

This trained model can also be invoked on new data. To do this use:

```bash
python3 task3_iam_test.py --input_dir <path to directory containing the images> --output <name of output txt file>
```
Don't include '.txt' in the output file name, this is already included in the code. The default name is 'output' and will be saved in the 'Task3' folder.

Use:

```bash
python3 task3_iam_test.py --help
```

to see the possible commands.
