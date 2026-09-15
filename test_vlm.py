import time
import logging
import torch
import numpy as np
from prismatic import load
from PIL import Image
import os
import cv2
import io
import base64
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from torchvision.transforms.functional import pil_to_tensor

from src.vlm import VLM
from am_radio import AMRadio
#from transformers import DetrImageProcessor, AutoModelForObjectDetections
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def detect_obj(ind, image, obj_name, color='r', save_dir=''):
    # Load model and processor
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

    # Load image
    image = Image.open(image).convert("RGB")

    # Natural language query
    #texts = [["a red backpack"]]  # You can put multiple phrases in a list if desired

    print("owl detection: ", obj_name)

    # Preprocess
    inputs = processor(text=obj_name, images=image, return_tensors="pt")

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Postprocess (get bounding boxes in pixel space)
    target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.01)

    # Get boxes and scores
    boxes = results[0]["boxes"]
    scores = results[0]["scores"]
    labels = results[0]["labels"]

    # Sort by highest score
    sorted_indices = scores.argsort(descending=True)

    # Apply sorting
    boxes = boxes[sorted_indices]
    scores = scores[sorted_indices]
    labels = labels[sorted_indices]

    # Print + visualize
    for box, score, label in zip(boxes, scores, labels):
        print(f"Detected '{obj_name}' with confidence {score:.2f} at {box.tolist()}")

        # Draw on image
        draw = image.copy()
        draw_box = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        plt.close('all')
        plt.imshow(draw)
        ax_o = plt.gca()
        ax_o.add_patch(draw_box)
        plt.title(f"{obj_name}: {score:.2f}")
        plt.axis("off")
    plt.savefig(os.path.join(save_dir, f"owl_det_{ind}.png"))
    plt.close()

        #return box
    
    return None

def query_doorway():
    img_path = 'results/vlm_explore/192-coverage/37.png'
    prompt = 'Am I currently looking at an open doorway?'

    vlm = VLM(cfg.vlm)
    res = vlm.generate(prompt, img)
    print("OUTPUT: ", res)

def pil_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# TEST grounded SAM detection
def query_gpt(prompt, images, T=0.4, max_tokens=512):
    
    #base64_image = self._pil_to_base64(image)
    image_messages = [{
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{pil_to_base64(img)}"
        }
    } for img in images]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                    
                ] + image_messages
            }
        ],
    )

    ret = response.choices[0].message.content
    print("gpt ret: ", ret)
    return ret

def query_groundingdino(image, text=''):
    model_id = "IDEA-Research/grounding-dino-base"
    processor = DetrImageProcessor.from_pretrained(model_id)
    model = AutoModelForObjectDetection.from_pretrained(model_id)
    text = "lamp next to the bed"
    #Preprocess
    inputs = processor(images=image, text=text, return_tensors="pt")

    # Forward pass
    outputs = model(**inputs)

    # Filter predictions
    target_sizes = torch.tensor([image.size[::-1]])  # H, W
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, target_sizes=target_sizes, box_threshold=0.3, text_threshold=0.25
    )[0]

    # Visualize
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
        x0, y0, x1, y1 = box
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x0, y0 - 5, f"{label} ({score:.2f})", color='red', fontsize=10)

    plt.axis("off")
    plt.save("gdino_image.png")

def test_gpt_qa():
    question = 'Is the lamp next to the bed on? \nA) (Do not choose this option) \nB) Yes, it is on \nC) No, it is not on \nD) (Do not choose this option).'
    question = 'Is the lamp next to the sofa turned on or off?\nA. (Do not choose this option)\nB. Off\nC. On\nD. (Do not choose this option)'
    prompt = f'''You are an agent exploring an environment to answer the question: {question}. Here are the most relevant images you've seen.
    Based on what you see in the image, what choice would you choose? Think step by step. Give the final answer as ANSWER: followed by the letter'''
    
    img_path = 'results/vlm_explore/0/'
    image_paths = ['results/vlm_explore/5/48.png'] #['results/vlm_explore/0/37.png', 'results/vlm_explore/0/43.png']
    images = []
    for path in image_paths:
        img = Image.open(path)
        images.append(img)

    query_gpt(prompt, images)

def test_dark_img():
    path = 'results/vlm_explore/136/17.png'
    rgb_im = np.array(Image.open(path).convert('RGB'))
    print(rgb_im)
    num_black_pixels = np.sum(
        np.sum(rgb_im, axis=-1) == 0
    )  # sum over channel first

    if num_black_pixels < 0.5 * 640 * 480:
        print("valid image")
    else:
        print("backn image")

def test_gpt_room():
    question = "Is the gray curtain pulled down in the bedroom with two twin beds?"
    prompt = '''You are an agent exploring a scene. What room are you most likely in at the moment? Only answer with the room name of a common room to have in a house.'''
    dir_prompt = f'''You are an agent trying to answer the question {question} what direction should you go in in the next step? Choose from the letters on the image.'''
    img_dir = 'results/vlm_explore/13/'
    images = []
    for i in range(1, 27):
        print("Image: ", i)
        img_path = os.path.join(img_dir, str(i)+'.png')
        img = Image.open(img_path)
        ret = query_gpt(prompt, [img])
        ret = query_gpt(dir_prompt, [img])

        break

def test_radio():
    # Load model
    hf_repo = "nvidia/RADIO"  # Use RADIO-H (general-purpose VLM)
    model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True).eval().to("cuda:4")
    #tokenizer = AutoTokenizer.from_pretrained(hf_repo, trust_remote_code=True)
    image_processor = CLIPImageProcessor.from_pretrained(hf_repo)

    # Load and process image
    image = Image.open('results/vlm_explore/13/1.png').convert('RGB')
    pixel_values = image_processor(images=image, return_tensors='pt', do_resize=True).pixel_values.to("cuda:4")

    # Process text query
    queries = ["doorway", "door", "room", "bed"]
    #text_tokens = tokenizer(queries, return_tensors="pt", padding=True, truncation=True).to("cuda:4")

    # Forward pass
    with torch.no_grad():
        image_summary, image_features = model(pixel_values)         # image_features: [B, D]
        text_summary, text_features = model.get_text_features(**text_tokens)  # text_features: [N, D]

    # Compute cosine similarity
    similarity = torch.nn.functional.cosine_similarity(
        image_features / image_features.norm(dim=-1, keepdim=True),
        text_features / text_features.norm(dim=-1, keepdim=True),
        dim=-1
    )

    # Find best match
    best_idx = similarity.argmax().item()
    print(f"Most relevant query: {queries[best_idx]} (score: {similarity[best_idx].item():.2f})")

def featurize():
    pil_image = Image.open('results/vlm_explore/192/17.png').convert('RGB')
    image = pil_to_tensor(pil_image).to(dtype=torch.float32, device='cuda')
    
    featurizer = AMRadio(adaptor_names="siglip")
    text = "entrance"
    heatmap = featurizer.text_alignment(text, image)

    contours, _ = featurizer.segment(heatmap)

    # Step 3 (Optional): Draw contours or bounding boxes on image
        # Assume original_image is shape (480, 640, 3)
    vis_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR) #image.detach().cpu().numpy()
    print(vis_image.shape)
    cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)

    # Or draw bounding boxes:
    #for cnt in contours:
    #    x, y, w, h = cv2.boundingRect(cnt)
    #    cv2.rectangle(vis_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
    # Show result
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title(f"Segmented regions: {text}")
    #plt.show()
    plt.savefig("17_segmented_"+text+'.png')

    # Show heatmap
    plt.imshow(heatmap, cmap='viridis')  # or 'hot', 'plasma', etc.
    plt.colorbar()
    plt.title("Lang-Aligned Similarity Heatmap")
    plt.axis("off")
    #plt.show()
    plt.savefig("17_siglip_feats_"+text+".png")

def room_classify():
    pil_image = Image.open('results/vlm_explore/192/38.png').convert('RGB')
    image = pil_to_tensor(pil_image).to(dtype=torch.float32, device='cuda')
    
    featurizer = AMRadio(adaptor_names="siglip")
    text = "bedroom"
    rooms = ['bedroom', 'living room', 'kitchen', 'kitchen', 'dining room', 'office', 'gym', 'laundry room', 'bathroom', 'garage', 'hallway']
    #heatmap = featurizer.text_alignment(text, image)
    pred_map = featurizer.room_alignment(rooms, image)

    # Define label -> color mapping
    colors = plt.cm.tab10(np.linspace(0, 1, len(rooms)))
    color_map = mcolors.ListedColormap(colors)

    plt.imshow(pred_map, cmap=color_map)
    plt.colorbar(ticks=range(len(rooms)), label='Room')
    plt.clim(-0.5, len(rooms) - 0.5)
    plt.title("Predicted Room per Patch")
    #plt.show()
    plt.savefig("38_room_class.png")

#def detect_objects():


if __name__ == "__main__":
    import argparse
    #from omegaconf import OmegaConf

    # get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="", type=str)
    args = parser.parse_args()
    
    # test_gpt_qa()
    #test_dark_img()
    #test_gpt_room()
    #test_radio()
    #featurize()
    #room_classify()
    detect_obj(34, 'results/vlm_explore/252/34.png', 'chair')