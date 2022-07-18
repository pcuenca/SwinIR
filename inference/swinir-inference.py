import argparse
import cv2
import itertools
import numpy as np
import torch
from datetime import datetime
from models.network_swinir import SwinIR as net
from pathlib import Path

tile = 256
device = 'cuda'
upscale = 4

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=30, help='batch size')
args = parser.parse_args()

bs = args.batch_size

images = Path('testsets/dalle-mini-samples/')

torch.set_grad_enabled(False)
#torch.set_num_threads(16)

# use 'nearest+conv' to avoid block artifacts
model = net(
    upscale=upscale, in_chans=3, img_size=64, window_size=8,
    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
    num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
    upsampler='nearest+conv', resi_connection='1conv'
)
param_key_g = 'params_ema'

# Weights for the "small" model, we need to download the large ones.
model_path = '../model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth'
pretrained_model = torch.load(model_path)
model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

model = model.eval().to(device)

def read_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img = img[:, :, [2, 1, 0]]
    img = np.transpose(img, (2, 0, 1))  # HCW to CHW
    img = torch.from_numpy(img).float().unsqueeze(0)
    return img

all_images = None
for image_path, _ in zip(itertools.cycle(images.iterdir()), range(bs)):
    img = read_image(str(image_path))
    all_images = img if all_images is None else torch.vstack((all_images, img))

start_time = datetime.now()
all_images = all_images.to(device)
out_images = model(all_images)
out_images = out_images.cpu().numpy()
elapsed = (datetime.now() - start_time).total_seconds()
print(f"bs: {bs}, time: {elapsed}, per image: {elapsed / bs}")
