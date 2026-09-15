# Modified from code by Pujith Kachana, 2025

import argparse
import os
import cv2
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor
import einops
import numpy as np
from PIL import Image

from featurizer import Featurizer


AVAIlABLE_MODEL_VERSIONS = [
    "radio_v2.5-g",
    "radio_v2.5-h",
    "radio_v2.5-l",
    "radio_v2.5-b",
    "e-radio_v2"
]

AVAILABLE_ADAPTORS = [
    "clip",
    "siglip",
    "dino_v2",
    "sam"
]


class AMRadio(Featurizer):
    
    def __init__(
        self, 
        model_version="radio_v2.5-b", 
        adaptor_names=None,
        device='cuda'
    ):
        super().__init__()
        self.device = device
        self.model_version = model_version
        self.adaptor_names = adaptor_names
        
        if self.model_version not in AVAIlABLE_MODEL_VERSIONS:
            raise ValueError(f"Model version {self.model_version} not available. Choose from {AVAIlABLE_MODEL_VERSIONS}")
        
        if self.adaptor_names is not None and self.adaptor_names not in AVAILABLE_ADAPTORS:
            raise ValueError(f"Adaptor {self.adaptor_names} not available. Choose from {AVAILABLE_ADAPTORS}")
            
        self.model = self.init_model()

        if self.adaptor_names is not None:
            self.lang_adaptor = self.model.adaptors[self.adaptor_names]


    def init_model(self):
        model = torch.hub.load(
            'NVlabs/RADIO', 
            'radio_model', 
            version=self.model_version, 
            adaptor_names=self.adaptor_names,
            progress=True, 
            skip_validation=True
        )
        model.to(self.device).eval()
        return model


    def process_images(self, images):
        images = images/255.0   # RADIO expects the input values to be between 0 and 1

        nearest_res = self.model.get_nearest_supported_resolution(*images.shape[-2:])
        images = F.interpolate(images, nearest_res, mode='bilinear', align_corners=False)

        if "e-radio" in self.model_version:
            self.model.model.set_optimal_window_size(x.shape[2:]) # where it expects a tuple of (height, width) of the input image.
        
        return images
    

    def featurize(self, images):
        processed_images = self.process_images(images)
        # RADIO expects the input to have values between [0, 1]. It will automatically normalize them to have mean 0 std 1.
        output = self.model(processed_images, feature_fmt='NCHW')

        if self.adaptor_names is None:
            summary, spatial_features = output
            return summary, spatial_features
        
        bb_summary, bb_features = output['backbone']
        lang_summary, lang_features = output[self.adaptor_names]

        b, c, h, w = bb_features.shape

        flattened_tokens = einops.rearrange(bb_features, 'b c h w -> (b h w) c')
        mlp_out = self.lang_adaptor.head_mlp(flattened_tokens)
        lang_aligned_feats = einops.rearrange(mlp_out, '(b h w) c -> b h w c', b=b, h=h, w=w)

        return lang_summary, lang_aligned_feats

    def text_alignment(self, text, images):
        images = images.unsqueeze(0)
        print(images.shape)

        summary, feats = self.featurize(images)
        print("feats shape: ", feats.shape)

        tokenizer = self.lang_adaptor.tokenizer
        #text = "doorway"
        tokens = tokenizer(text).to(device='cuda')
        text_feats = self.lang_adaptor.encode_text(tokens, normalize=True)
        #text_feats = text_feats.detach().cpu()

        print("text_feats shape: ", text_feats.shape)

        image_features = feats.squeeze(0)       # [30, 40, 1152]
        text_feature = text_feats.squeeze(0)           # [1152]

        # Normalize both
        image_features = torch.nn.functional.normalize(image_features, dim=-1)  # [30, 40, 1152]
        text_feature = torch.nn.functional.normalize(text_feature, dim=0)       # [1152]

        # Compute cosine similarity
        # This will broadcast text_feature to [30, 40, 1152] and compute dot product
        similarity_map = (image_features @ text_feature)  # [30, 40]

        # Convert to numpy
        sim_np = similarity_map.detach().cpu().numpy()

        # Normalize
        sim_np = (sim_np - sim_np.min()) / (sim_np.max() - sim_np.min())

        heatmap = cv2.resize(sim_np, (640, 480), interpolation=cv2.INTER_CUBIC)

        return heatmap

    def segment(self, heatmap, thresh=0.75):

        binary_mask = (heatmap > thresh).astype(np.uint8)  # 0s and 1s

        # morphological cleaning (remove noise)
        kernel = np.ones((10, 10), np.uint8)
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        # find contours
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = 500  # adjust based on image resolution
        filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        #print(filtered_contours)
        print(len(filtered_contours))

        if len(filtered_contours) > 0:
            sorted_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)

            largest = max(filtered_contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            max_centroid = (heatmap.shape[0]/2, heatmap.shape[1]/2)
            if M["m00"] != 0:  # to avoid division by zero
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Draw the center point
                #cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
                max_centroid = (cX, cY)
            return sorted_contours, max_centroid
        else:
            return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True, help="Path to directory containing images")
    args = parser.parse_args()

    images = [Image.open(os.path.join(args.img_dir, img)).convert('RGB') for img in os.listdir(args.img_dir)]
    images = torch.stack([pil_to_tensor(img) for img in images]).to(dtype=torch.float32, device='cuda')   

    featurizer = AMRadio(adaptor_names="siglip")
    summary, spatial_features = featurizer.featurize(images)


    