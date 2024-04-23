import os
import re
import cv2
import torch
import shutil
import argparse
import warnings
from models import *
from tqdm import tqdm
from collections import OrderedDict
from datasets.loader import PairLoader
from torch.utils.data import DataLoader
from utils import AverageMeter, write_img, chw_to_hwc

def delete_directory_contents(directory_path):
    try:
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            elif os.path.isfile(item_path):
                os.remove(item_path)
            else:
                print(f"Unknown Files : {item_path}")
    except Exception as e:
        print(f"Unknown Error : {str(e)}")

def video_to_image(videoPath):
    cap = cv2.VideoCapture(videoPath)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _, frame = cap.read()

    writePath = "/hazy"
    os.makedirs(writePath, exist_ok=True)
    
    for i in tqdm(range(length)):
        cv2.imwrite(writePath + "/%06d.jpg" % i, frame)
        _, frame = cap.read()

    cap.release()

def dehaze(network, args):
	dataset_dir = "/hazy"
	image = cv2.imread(dataset_dir + "/000000.jpg")
	h, w = image.shape[:2]
	haze_dataset = PairLoader(dataset_dir, 'valid', size = [h, w])

	test_loader = DataLoader(haze_dataset,
							 batch_size=1,
							 num_workers=args.num_workers,
							 pin_memory=True)
	network.eval()

	result_dir = "/dehaze"
	os.makedirs(os.path.join(result_dir), exist_ok=True)

	for idx, batch in enumerate(tqdm(test_loader)):
		input = batch['source'].cuda()
		target = batch['target'].cuda()
        
		filename = batch['filename'][0]
		result_file = os.path.join(result_dir, filename)
          
		with torch.no_grad():
			output = network(input).clamp_(-1, 1)
			output = output * 0.5 + 0.5
			target = target * 0.5 + 0.5

		out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
		write_img(result_file, out_img)
		torch.cuda.empty_cache()

def image_to_video(args, fps):
    output_path = "./output/"
    writePath = os.path.join(output_path, args.o)
    directory = os.path.dirname(writePath)
    os.makedirs(directory, exist_ok=True)

    filepath = "/dehaze"
    filenames = os.listdir(filepath)
    filenames.sort()

    image = cv2.imread(filepath + "/" + filenames[0])
    h, w = image.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(writePath, fourcc, fps, (w, h))
    
	
    for filename in tqdm(filenames):
        img = cv2.imread(filepath + "/" + filename)
        out.write(img)
	
    out.release()
    delete_directory_contents("/dehaze")
    delete_directory_contents("/hazy")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', default='./files/test.mp4', type=str, help='Path of a Hazy video File.')
    parser.add_argument('-o', default='test.mp4', type=str, help='Path of Output files.')
    parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
    args = parser.parse_args()

    network = eval("dehazeformer")()
    network.cuda()

    saved_model_dir = "./saved_models/dehazeformer.pth"

    if os.path.exists(saved_model_dir):
        network.load_state_dict(torch.load(saved_model_dir, map_location=torch.device('cuda'))['state_dict'])
    else:
        print('==> No existing trained model!')
        exit(0)
    
    if not os.path.exists(args.i):
        print('==> No existing hazy video!')
        exit(0)

    cap = cv2.VideoCapture(args.i)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print('==> Extracting Frames')
    video_to_image(args.i)
    print('==> Start Dehazing...')
    dehaze(network, args)
    print('==> Merging...')
    image_to_video(args, fps)