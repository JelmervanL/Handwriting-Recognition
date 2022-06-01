import sys
import os
from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor
from PIL import Image
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

if __name__ == '__main__':
	# Parse command line arguments
	parser = ArgumentParser(description = '', formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument("-i", "--input_dir", help="Path to input that contains the line images")
	parser.add_argument("-o", "--output_dir", default='results/', help="output directory where the txt file for each image will be saved")
	
	args = vars(parser.parse_args())

	img_dir = args["input_dir"]
	results_filepath = args["output_dir"]

	# create results directory for output
	if not os.path.exists(results_filepath):
	    os.makedirs(results_filepath)

	print("Input image directory:", img_dir)
	print("Results will be saved in:", results_filepath)

	print("Loading model...")
	# load our finetuned model
	model = VisionEncoderDecoderModel.from_pretrained("models/trocr-small-5beam/checkpoint-3730", local_files_only=True)
	# load pretrained processor
	processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")

	for file in os.listdir(img_dir):
		# get predicition
		image = Image.open(os.path.join(img_dir, file)).convert("RGB")
		pixel_values = processor(image, return_tensors="pt").pixel_values
		generated_ids = model.generate(pixel_values)
		generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

		print(file)
		print(generated_text)
		print("\n")

		image_name = file.split('.')[0]

		# write to txt file
		txt_file_savepath = os.path.join(results_filepath, image_name + '_characters.txt')
		with open(txt_file_savepath, 'w') as txt_file:
			txt_file.write(generated_text + '\n')

	print("##### Finished #####")