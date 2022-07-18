import cv2
import itertools
import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from datetime import datetime
from models.network_swinir import SwinIR as net
from pathlib import Path

bs = 2      # Per core
iterations = 50

upscale = 4

torch.set_grad_enabled(False)
torch.set_num_threads(1)

def load_model():
    model = net(
        upscale=upscale, in_chans=3, img_size=64, window_size=8,
        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv'
    )
    model_path = 'model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth'
    param_key_g = 'params_ema'

    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    return model

def read_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img = img[:, :, [2, 1, 0]]
    img = np.transpose(img, (2, 0, 1))  # HCW to CHW
    img = torch.from_numpy(img).float().unsqueeze(0)
    return img

def _mp_fn(index):
    device = xm.xla_device()

    model = load_model().eval().to(device)

    images = Path('testsets/dalle-mini-samples/')

    num_images = bs * iterations
    batch = None
    for image_path, num in zip(itertools.cycle(images.iterdir()), range(num_images)):
        img = read_image(str(image_path))
        batch = img if batch is None else torch.vstack((batch, img))
        if num % bs == 0:
            start_time = datetime.now()
            batch = batch.to(device)
            out_images = model(batch)
            out_images = out_images.cpu().numpy()
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"Batch: {elapsed}, device: {index}")
            batch = None

if __name__ == '__main__':
        xmp.spawn(_mp_fn, args=(), start_method='fork')
