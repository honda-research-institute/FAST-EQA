import torch
from PIL import Image
import time
import logging
from types import SimpleNamespace
from transformers import CLIPProcessor, CLIPModel
from argparse import ArgumentParser
from functools import lru_cache

def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

# Simple text embedding cache
@lru_cache(maxsize=4096)
def _cached_text_embed(text_tuple, device_str):
    # text_tuple is a tuple to make hashable
    return device_str  # placeholder; cache stores below via wrapper

def get_clip_rel_fast(model, processor, text, image=None, device=None, fp16=True):
    start = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    amp_dtype = torch.float16 if (fp16 and device == "cuda") else None
    autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype else torch.autocast("cpu", enabled=False)

    with torch.inference_mode(), autocast_ctx:
        if image is not None:
            # image and text
            # move inputs to device
            inputs = processor(text=text, images=image, return_tensors="pt", padding=True, truncation=True)
            inputs = to_device(inputs, device)

            outputs = model(**inputs)
            image_embeds = outputs.image_embeds  # [B, D]
            text_embeds  = outputs.text_embeds   # [B, D]

            # Normalize
            image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)
            text_embeds  = torch.nn.functional.normalize(text_embeds,  dim=-1)

            # Cosine similarity
            similarity = (image_embeds @ text_embeds.T)
        else:
            # comparing between two texts
            t1, t2 = text

            # Use tokenizer-only path (faster than processor with images=None)
            tok1 = processor.tokenizer([t1], return_tensors="pt", padding=True, truncation=True)
            tok2 = processor.tokenizer([t2], return_tensors="pt", padding=True, truncation=True)
            tok1 = to_device(tok1, device)
            tok2 = to_device(tok2, device)

            e1 = model.get_text_features(**tok1)  # [1, D]
            e2 = model.get_text_features(**tok2)  # [1, D]

            # Normalize
            e1 = torch.nn.functional.normalize(e1, dim=-1)
            e2 = torch.nn.functional.normalize(e2, dim=-1)

            # Cosine similarity
            similarity = torch.nn.functional.cosine_similarity(e1, e2).item()
        
        end = time.time()
        logging.info(f"CLIP query time: {end-start}")
        return similarity

def get_clip_rel(model, processor, text, image=None):
    start = time.time()
    # Preprocess
    if image:
        inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds  # shape: [1, 512]
            text_embeds = outputs.text_embeds    # shape: [1, 512]
        
        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        print("text_embeds shape", text_embeds.shape)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        similarity = (image_embeds @ text_embeds.T)
    else:
        inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)

        # Normalize and compute cosine similarity
        caption_emb = outputs[0] / outputs[0].norm()
        object_emb = outputs[1] / outputs[1].norm()
        similarity = torch.nn.functional.cosine_similarity(caption_emb, object_emb, dim=0).item()

    #print(similarity)
    end = time.time()
    logging.info(f"CLIP query time: {end-start}")
    return similarity


if __name__ == "__main__":
    import argparse
    # test inference time

    # get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="", type=str)
    args = parser.parse_args()

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float16 if torch.cuda.is_available() else None)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    visual_goals = ['white object above the TV', 'dining table']
    rgb_im = Image.open('results/vlm_hmeqa/0/0.png').convert('RGB')

    print("Querying CLIP regular")
    sim = get_clip_rel(clip_model, processor, visual_goals + ["doorway to another room"], rgb_im)
    print(sim)

    print("\nQuerying faster implementation")
    sim = get_clip_rel_fast(clip_model, processor, visual_goals + ["doorway to another room"], rgb_im)
    print(sim)