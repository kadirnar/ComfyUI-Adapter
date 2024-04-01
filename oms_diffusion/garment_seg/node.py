import comfyui
import torch
import nodes

from oms_diffusion.garment_seg.network import U2NET
from oms_diffusion.utils.image_utils import prepare_image, prepare_mask

import torchvision.transforms as transforms

def load_checkpoint(model, checkpoint_path):
    from collections import OrderedDict

    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print("----checkpoints loaded from path: {}----".format(checkpoint_path))
    return model


class Normalize_image(object):
    """Normalize given tensor into given mean and standard dev

    Args:
        mean (float): Desired mean to substract from tensors
        std (float): Desired std to divide from tensors
    """

    def __init__(self, mean, std):
        assert isinstance(mean, (float))
        if isinstance(mean, float):
            self.mean = mean

        if isinstance(std, float):
            self.std = std

        self.normalize_1 = transforms.Normalize(self.mean, self.std)
        self.normalize_3 = transforms.Normalize([self.mean] * 3, [self.std] * 3)
        self.normalize_18 = transforms.Normalize([self.mean] * 18, [self.std] * 18)

    def __call__(self, image_tensor):
        if image_tensor.shape[0] == 1:
            return self.normalize_1(image_tensor)

        elif image_tensor.shape[0] == 3:
            return self.normalize_3(image_tensor)

        elif image_tensor.shape[0] == 18:
            return self.normalize_18(image_tensor)

        else:
            assert "Please set proper channels! Normlization implemented only for 1, 3 and 18"


def apply_transform(img):
    transforms_list = []
    transforms_list += [transforms.ToTensor()]
    transforms_list += [Normalize_image(0.5, 0.5)]
    transform_rgb = transforms.Compose(transforms_list)
    return transform_rgb(img)


def generate_mask(input_image, net, device='cpu'):
    import torch.nn.functional as F
    from PIL import Image
    import numpy as np


    img = input_image
    img_size = img.size
    img = img.resize((768, 768), Image.BICUBIC)
    image_tensor = apply_transform(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    with torch.no_grad():
        output_tensor = net(image_tensor.to(device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()
        mask = (output_arr != 0).astype(np.uint8) * 255
        mask = mask[0]  # Selecting the first channel to make it 2D
        alpha_mask_img = Image.fromarray(mask, mode='L')
        alpha_mask_img = alpha_mask_img.resize(img_size, Image.BICUBIC)

    return alpha_mask_img

def load_checkpoint(model, checkpoint_path):
    from collections import OrderedDict

    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print("----checkpoints loaded from path: {}----".format(checkpoint_path))
    return model


def load_seg_model(checkpoint_path, device='cpu'):
    net = U2NET(in_ch=3, out_ch=4)
    net = load_checkpoint(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()

    return net

class GarmentSegLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cloth_image": ("IMAGE",),
                "seg_model_path": ("STRING",),
                "device": ("STRING",),
                "height": ("INT",),
                "width": ("INT",),
            },
        }
    
    
    RETURN_TYPES = ("",)
    FUNCTION = ""
    CATEGORY = "ComfyUI-Adapter/OmsDiffusion/GarmentSeg"
    

    def garment_seg_loader(self, cloth_image, seg_model_path, device, height, width):
        segment_model = load_seg_model(seg_model_path, device=device)
        cloth_mask_image = generate_mask(cloth_image, net=segment_model, device=device)
        cloth = prepare_image(cloth_image, height, width)
        cloth_mask = prepare_mask(cloth_mask_image, height, width)
        cloth = (cloth * cloth_mask).to(self.device, dtype=torch.float16)

