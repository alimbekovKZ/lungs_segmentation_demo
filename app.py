"""Streamlit web app for lungs segmentation"""

from collections import namedtuple

import albumentations as A
import cv2
import numpy as np
import streamlit as st
import torch
from torch import nn
from torch.utils import model_zoo

import unet as Unet

st.set_option("deprecation.showfileUploaderEncoding", False)

img_size = 512
aug = A.Compose([A.Resize(img_size, img_size, interpolation=1, p=1)], p=1)

model = namedtuple("model", ["url", "model"])

models = {
    "resnet34": model(
        url="https://github.com/alimbekovKZ/lungs_segmentation/releases/download/1.0.0/resnet34.pth",
        model=Unet.Resnet(seg_classes=2, backbone_arch="resnet34"),
    ),
    "densenet121": model(
        url="https://github.com/alimbekovKZ/lungs_segmentation/releases/download/1.0.0/densenet121.pth",
        model=Unet.DensenetUnet(seg_classes=2, backbone_arch="densenet121"),
    ),
}


def create_model(model_name: str) -> nn.Module:
    model = models[model_name].model
    state_dict = model_zoo.load_url(
        models[model_name].url, progress=True, map_location="cpu"
    )
    model.load_state_dict(state_dict)
    return model


@st.cache(allow_output_mutation=True)
def cached_model():
    model = create_model("resnet34")
    device = torch.device("cpu")
    model = model.to(device)
    return model


model = cached_model()

st.title("Segment lungs")

# uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

file = st.sidebar.file_uploader(
    "Upload your image (jpg, jpeg, or png)", ["jpg", "jpeg", "png"]
)


def img_with_masks(img, masks, alpha, return_colors=False):
    """
    returns image with masks,
    img - numpy array of image
    masks - list of masks. Maximum 6 masks. only 0 and 1 allowed
    alpha - int transparency [0:1]
    return_colors returns list of names of colors of each mask
    """
    colors = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [102, 51, 0],
    ]
    color_names = ["Red", "greed", "BLue", "Yello", "Light", "Brown"]
    img = img - img.min()
    img = img / (img.max() - img.min())
    img *= 255
    img = img.astype(np.uint8)

    c = 0
    for mask in masks:
        mask = np.dstack((mask, mask, mask)) * np.array(colors[c])
        mask = mask.astype(np.uint8)
        img = cv2.addWeighted(mask, alpha, img, 1, 0.0)
        c = c + 1
    if return_colors is False:
        return img
    else:
        return img, color_names[0 : len(masks)]


def inference(model, image, thresh=0.2):
    model.eval()
    image = (image - image.min()) / (image.max() - image.min())
    augs = aug(image=image)
    image = augs["image"].transpose((2, 0, 1))
    im = augs["image"]
    image = np.expand_dims(image, axis=0)
    image = torch.tensor(image)

    mask = torch.nn.Sigmoid()(model(image.float()))
    mask = mask[0, :, :, :].cpu().detach().numpy()
    mask = (mask > thresh).astype("uint8")
    return im, mask


if file is not None:
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption="Before", use_column_width=True)
    st.write("")
    st.write("Detecting lungs...")
    image, mask = inference(model, image, 0.2)
    st.image(
        img_with_masks(image, [mask[0], mask[1]], alpha=0.1),
        caption="Image + mask",
        use_column_width=True,
    )
