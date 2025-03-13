from PIL import Image

from fastsam import FastSAM, FastSAMPrompt
import supervision as sv
import argparse

model = FastSAM('./weights/FastSAM-x.pt')
IMAGE_PATH = 'images/dog.jpg'
OUTPUT_IMAGE_PATH = 'output/output.jpg'
DEVICE = 'cuda'

everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.7)
prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

# types
# 1 - Tout segmenter
# 2 - Segmenter via un texte
# input - Texte pour la segmentation
# coordinates - Coordonnées pour la segmentation (ex: 50:50,120:80)
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", type=int)
parser.add_argument("-i", "--input", type=str)
parser.add_argument("-c", "--coordinates", type=str)
args = parser.parse_args()

if args.type == 1:
    ann = prompt_process.everything_prompt()
elif args.type == 2:
    ann = prompt_process.text_prompt(args.input)
elif args.type == 3:
    coordinates = []
    pointLabels = []
    coordinates_args = args.coordinates.split(',')
    for coordinates_arg in coordinates_args:
        coordinates_number = coordinates_arg.split(':')
        coordinates.append([int(coordinates_number[0]), int(coordinates_number[1])])
        pointLabels.append(1)

    ann = prompt_process.point_prompt(coordinates, pointLabels)

prompt_process.plot(annotations=ann,output_path=OUTPUT_IMAGE_PATH)

image = Image.open(OUTPUT_IMAGE_PATH)
sv.plot_image(image)
