import torch
import random
from torchvision.transforms import functional as F
from torchvision import transforms


class MVTecTransformWithMaskTest(object):
    """
    Apply the same augmentations to the mask as to the original image from test dataset.
    """

    def __init__(self, image_size, crop_pct, interpolation, mean, std):
        self.image_size = image_size
        self.crop_pct = crop_pct
        self.interpolation = interpolation
        self.mean = mean
        self.std = std

    def __call__(self, image, mask=None):
        img_t = F.resize(image, size=int(self.image_size/self.crop_pct))
        if mask is not None:
            mask_t = F.resize(mask, size=int(self.image_size/self.crop_pct))
        
        k = (int(self.image_size/self.crop_pct) - self.image_size)//2
        crop_params = (k, k, self.image_size, self.image_size)

        img_t = F.crop(img_t, *crop_params)
        if mask is not None:
            mask_t = F.crop(mask_t, *crop_params)

        img_t = F.to_tensor(img_t)
        if mask is not None:
            mask_t = F.to_tensor(mask_t)

        img_t = F.normalize(img_t, mean=self.mean, std=self.std)

        if mask is not None:
            return img_t, mask_t
        else:
            return img_t, mask
        

class MVTecTransformWithMaskTrain(object):
    """
    Apply the same augmentations to the mask as to the original image from train dataset.
    Contains: random crop, hflip, vflip, colorjitter, rotate, adjust_sharpness, gaussian blur and posterize.
    """

    def __init__(self, image_size, crop_pct, interpolation, mean, std):
        self.image_size = image_size
        self.crop_pct = crop_pct
        self.interpolation = interpolation
        self.mean = mean
        self.std = std

    def __call__(self, image, mask=None):
        # random crop
        img_t = F.resize(image, size=int(self.image_size/self.crop_pct))
        if mask is not None:
            mask_t = F.resize(mask, size=int(self.image_size/self.crop_pct))

        k = int(self.image_size/self.crop_pct) - self.image_size
        crop_random_x = int(torch.rand(1)*k)
        crop_random_y = int(torch.rand(1)*k)

        crop_params = (crop_random_x, crop_random_y, self.image_size, self.image_size)

        img_t = F.crop(img_t, *crop_params)
        if mask is not None:
            mask_t = F.crop(mask_t, *crop_params)

        # hflip (mask)
        hfilp_random = torch.rand(1)
        if hfilp_random<0.5:
            img_t = F.hflip(img_t)
            if mask is not None:
                mask_t = F.hflip(mask_t)
        

        # vflip (mask)
        vfilp_random = torch.rand(1)
        if vfilp_random<0.5:
            img_t = F.vflip(img_t)
            if mask is not None:
                mask_t = F.vflip(mask_t)

        # colorjitter 
        img_t = transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.8))(img_t)

        # rotate (mask)
        rotate_random = torch.rand(1)
        if rotate_random < 0.25:
            img_t = F.rotate(img_t, 90)
            if mask is not None:
                mask_t = F.rotate(mask_t, 90)
        elif rotate_random < 0.5:
            img_t = F.rotate(img_t, 180)
            if mask is not None:
                mask_t = F.rotate(mask_t, 180)
        elif rotate_random < 0.75:
            img_t = F.rotate(img_t, 270)
            if mask is not None:
                mask_t = F.rotate(mask_t, 270)
        

        random_sharpness = random.uniform(0, 1)
        img_t = F.adjust_sharpness(img_t, 0.2+random_sharpness*1.8)

        # random gaussian
        random_kernel = random.choice([9,13,17,21,25,29,33])
        img_t = transforms.GaussianBlur(kernel_size=random_kernel)(img_t)

        # random posterize
        random_posterize = random.choice([5,6,7,8])
        img_t = F.posterize(img_t, random_posterize)

        img_t = F.to_tensor(img_t)
        if mask is not None:
            mask_t = F.to_tensor(mask_t)

        img_t = F.normalize(img_t, mean=self.mean, std=self.std)
        
        if mask is not None:
            return img_t, mask_t
        else:
            return img_t, mask
        

def get_transform(image_size=32, args=None):
    """
    Apply the same augmentations to the mask as to the original image
    """

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    interpolation = args.interpolation
    crop_pct = args.crop_pct

    train_transform = MVTecTransformWithMaskTrain(image_size, crop_pct, interpolation, mean, std)

    test_transform = MVTecTransformWithMaskTest(image_size, crop_pct, interpolation, mean, std)

    return (train_transform, test_transform)


    
class ContrastiveLearningViewGenerator(object):
    """
    Take two random crops of one image and its mask as the query and key.
    """
    
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, image, mask):
        if not isinstance(self.base_transform, list):
            transformed_images = []
            transformed_masks = []
            for i in range(self.n_views): 
                transformed_image, transformed_mask =  self.base_transform(image, mask)
                transformed_images.append(transformed_image)
                transformed_masks.append(transformed_mask)
            return transformed_images, transformed_masks
        else:
            raise NotImplementedError