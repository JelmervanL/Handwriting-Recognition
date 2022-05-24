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
	parser.add_argument("-o", "--output", default='output', help="Name of output txt file with image names and text predictions")
	
	args = vars(parser.parse_args())

	img_dir = args["input_dir"]
	output_file = args["output"]

	print("input image directory:", img_dir)

	print("Loading model...")
	# load finetuned model; Have to change to our pretrained model
	model = VisionEncoderDecoderModel.from_pretrained("models/trocr-small-5beam/checkpoint-3730", local_files_only=True)
	# load pretrained processor
	processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")

	txt_file = open(output_file + '.txt', 'w')
	for file in os.listdir(img_dir):
		image = Image.open(os.path.join(img_dir, file)).convert("RGB")
		pixel_values = processor(image, return_tensors="pt").pixel_values
		generated_ids = model.generate(pixel_values)
		generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
		print(file)
		print(generated_text)
		txt_file.write(file + '\n')
		txt_file.write(generated_text + '\n')
		txt_file.write('\n')

	txt_file.close()

