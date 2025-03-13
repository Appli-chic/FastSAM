from fastsam import FastSAM, FastSAMPrompt

model = FastSAM('./weights/FastSAM-x.pt')
IMAGE_PATH = 'images/dog.jpg'
DEVICE = 'cuda'
everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.7)
prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

ann = prompt_process.everything_prompt()

prompt_process.plot(annotations=ann,output_path='./output/output.jpg')
