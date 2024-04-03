import torch
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
from pathlib import Path
from .network import U2NET

from huggingface_hub import snapshot_download

base_path = os.path.join(Path(__file__).absolute().parents[2].absolute())
current_paths = os.path.join(base_path, "checkpoints")

# Check if the 'checkpoints' directory exists
if not os.path.exists(current_paths):
    os.makedirs(current_paths)
    print(f"'{current_paths}' directory created successfully.")
else:
    print(f"'{current_paths}' directory already exists.")

pth_files = [file for file in os.listdir(current_paths) if file.endswith(".pth")]

if not pth_files:
    print("No .pth files found in the 'checkpoints' directory.")

    snapshot_download(
        repo_id="TryOnVirtual/ClothSeg", 
        repo_type="model", 
        ignore_patterns=["*.md", "*.gitattributes"],
        local_dir=current_paths,
)

else:
    print(".pth files found in the 'checkpoints' directory. Skipping snapshot download.")
    
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

class GarmentSeg:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cloth_tensor_image": ("IMAGE",{"default":None}),
            },
        }
    
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "garment_seg_loader"
    CATEGORY = "ComfyUI-Adapter/OmsDiffusion/GarmentSeg"
    

    def garment_seg_loader(self, cloth_tensor_image):
        seg_model_path = os.path.join(current_paths, "cloth_segm.pth")

        image = (cloth_tensor_image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)
        
        segment_model = load_seg_model(seg_model_path)
        
        cloth_mask_image = generate_mask(pil_image, net=segment_model)
        npy_cloth_mask_image = np.array(cloth_mask_image).astype(np.float32) / 255.0
        
        tensor_cloth_mask_image = torch.tensor(npy_cloth_mask_image).unsqueeze(0)

        return (tensor_cloth_mask_image,)