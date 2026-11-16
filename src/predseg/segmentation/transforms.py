import torchvision
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


_IMAGE_MEAN = [0.485, 0.456, 0.406]
_IMAGE_STD = [0.229, 0.224, 0.225]


class SegmentationPresetTrain:
    def __init__(self, crop_size=224):
        self.crop_size = crop_size

    def __call__(self, image: Image.Image, mask: Image.Image):
        top, left, height, width = torchvision.transforms.RandomResizedCrop.get_params(
            image,
            scale=(0.95, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
        )
        output_size = [self.crop_size, self.crop_size]
        image = F.resized_crop(
            image,
            top,
            left,
            height,
            width,
            output_size,
            interpolation=InterpolationMode.BILINEAR,
        )
        mask = F.resized_crop(
            mask,
            top,
            left,
            height,
            width,
            output_size,
            interpolation=InterpolationMode.NEAREST,
        )
        image = F.normalize(
            F.to_tensor(image),
            mean=_IMAGE_MEAN,
            std=_IMAGE_STD,
        )
        mask = (F.pil_to_tensor(mask) > 127).float()
        return image, mask


class SegmentationPresetEval:
    def __init__(self, crop_size=224):
        self.crop_size = crop_size

    def __call__(self, image: Image.Image, mask: Image.Image):
        output_size = [self.crop_size, self.crop_size]
        image = F.resize(
            image,
            output_size,
            interpolation=InterpolationMode.BILINEAR,
        )
        mask = F.resize(
            mask,
            output_size,
            interpolation=InterpolationMode.NEAREST,
        )
        image = F.normalize(
            F.to_tensor(image),
            mean=_IMAGE_MEAN,
            std=_IMAGE_STD,
        )
        mask = (F.pil_to_tensor(mask) > 127).float()
        return image, mask


def get_transform(train, crop_size=224):
    if train:
        return SegmentationPresetTrain(crop_size)
    return SegmentationPresetEval(crop_size)
