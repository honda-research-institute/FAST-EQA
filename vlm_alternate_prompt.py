import time
import logging
import torch
import traceback
import numpy as np
import os
import io
import json
import ast
import requests
import base64
from PIL import Image
from prismatic import load
import openai
import re
from mistralai import Mistral
from typing import Optional

openai.api_key = os.getenv("OPENAI_API_KEY")

###### NOTE ######
# this is an alternate version of the VLM() class in vlm.py that gets rooms and visual goals with one GPT call

# To replace current logic in run_vlm_explore.py for room extraction and visual goal extraction, using this alternate call
# use the code below for both single and multi-target:

'''
goals = vlm.extract_goals(question)
visual_goals = goals["visual_goals"]
rooms_to_explore = goals["rooms"]
logging.info(f"Rooms to explore: {rooms_to_explore}")
for goal in visual_goals:
    mt_clip_scores.append([])
    mt_combined_scores.append([])

logging.info(f"visual goal: {visual_goals}")
num_targets = len(visual_goals)
logging.info(f"num targets: {num_targets}")'''

# alternate VLM class
class VLM:
    def __init__(self, cfg):
        self.model_name = cfg.model_name
        if cfg.model_name == 'prismatic':
            start_time = time.time()
            self.model = load(cfg.model_id, hf_token=cfg.hf_token)
            self.model.to(cfg.device, dtype=torch.bfloat16)
            logging.info(f"Loaded VLM in {time.time() - start_time:.3f}s")
        elif cfg.model_name == 'gpt-4o':
            logging.info(f"Using {self.model_name} from API")
        
        self.mistral_key = os.getenv("MISTRAL_KEY")
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def _pil_to_base64(self, image):
        
        #print("type: ", type(image))
        if isinstance(image, np.ndarray):
            print(f"Image hsape: {image.shape}")
            print(f"squeezed: {np.squeeze(image).shape}")
            image = Image.fromarray(image.astype(np.uint8))
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def generate(self, prompt, images, T=0.4, max_tokens=512):
        if self.model_name == 'prismatic':
            prompt_builder = self.model.get_prompt_builder()
            prompt_builder.add_turn(role="human", message=prompt)
            prompt_text = prompt_builder.get_prompt()
            generated_text = self.model.generate(
                images,
                prompt_text,
                do_sample=True,
                temperature=T,
                max_new_tokens=max_tokens,
                min_length=1,
            )
            return generated_text
        elif self.model_name == 'gpt-4o':
            #base64_image = self._pil_to_base64(image)
            image_messages = [{
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{self._pil_to_base64(img)}"
                }
            } for img in images]
            response = self.client.chat.completions.create(
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

    def parse_mc_answer(self, response):

        match = re.search(r'ANSWER:\s*([A-Z])\b', response)
        if match:
            letter = match.group(1)
            print(f"Extracted answer: {letter}")
        else:
            print("No answer found.")
            return None
        
        return letter
    
    def parse_ov_answer(self, response):
        match = re.search(r'ANSWER:\s*(.*)', response, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            print(f"Extracted answer: {answer}")
            return answer
        else:
            print("No answer found.")
            return response.split('\n')[-1]
        
    def query_gpt(self, prompt, images, multichoice=False, T=0.4, max_tokens=512):
        image_messages = [{
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{self._pil_to_base64(img)}"
            }
        } for img in images]

        tries = 3
        while tries > 0:
            response = self.client.chat.completions.create(
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
            logging.info(f"gpt ret: {ret}")
            # retry if answer invalid
            if not multichoice:
                if ret is None:
                    tries -= 1
                else:
                    answer = self.parse_ov_answer(ret)
                    return answer
            else:
                choice = self.parse_mc_answer(ret)
                if choice is None:
                    tries -= 1
                else:
                    return choice
        return choice

    def get_loss(self, image, prompt, tokens, get_smx=True, T=1):
        "Get unnormalized losses (negative logits) of the tokens"
        prompt_builder = self.model.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=prompt)
        prompt_text = prompt_builder.get_prompt()
        losses = self.model.get_loss(
            image,
            prompt_text,
            return_string_probabilities=tokens,
        )[0]
        losses = np.array(losses)
        if get_smx:
            return np.exp(-losses / T) / np.sum(np.exp(-losses / T))
        return losses

    def parse_score(self, output: str, tag: str = "Your mark:") -> str:
        if output.isdigit():
            return int(output)
        start_idx = output.find(tag)
        if start_idx == -1:
            #raise ValueError("Invalid output string: {}".format(output))
            print("Invalid output string: {}".format(output))
            return None
        end_idx = output.find("\n", start_idx)
        if end_idx == -1:
            return int(output[start_idx:].replace(tag, "").strip())
        return int(output[start_idx:end_idx].replace(tag, "").strip())

    def parse_plan_json(self, output):
        # extracts task plan from LLM output in JSON format

        # Strip leading/trailing whitespace
        s = output.strip()

        # remove markdown style ```json```
        s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"```$", "", s).strip()

        # Remove surrounding quotes if present
        if (s.startswith("'''") and s.endswith("'''")) or (s.startswith('"""') and s.endswith('"""')):
            s = s[3:-3].strip()
        elif (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1].strip()

        # Try to parse the cleaned string
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            print("JSON parsing failed:", e)
            return None
    
    def parse_goals(self, output):
        # parses output from calling extract_goals where LLM outputs a JSON object
        s = output.strip()

        # check for fenced code block
        fence = re.search(r"```(?:json|javascript|js|py|python)?\s*(.*?)```", s, flags=re.IGNORECASE | re.DOTALL)
        if fence:
            s = fence.group(1).strip()

        #try to find a JSON object/array anywhere in the string
        decoder = json.JSONDecoder()
        for m in re.finditer(r"[{\[]", s):
            try:
                obj, end = decoder.raw_decode(s[m.start():])
                return obj
            except json.JSONDecodeError:
                continue

        # in case single quotes (these cases should be backup)
        try:
            obj = ast.literal_eval(s)
            return obj
        except Exception:
            pass
        
        brace = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not brace:
            brace = re.search(r"\[.*\]", s, flags=re.DOTALL)
        if brace:
            chunk = brace.group(0)
            try:
                return json.loads(chunk)
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(chunk)
                except Exception:
                    pass

        return None

    def extract_goals(self, question):
        # extract_goals function with new prompt to get rooms needed and visual_goals (targets) in one API call

        prompt = '''You are an agent tasked with exploring an environment to answer a question.
    First, given the question, output the detailed visual object goal(s) you need to observe in order to answer the question as a list of strings. There can be multiple. If no target object is mentioned, fill in 'object'.
    Please also give the relevant rooms you need to explore to see the goal object. Choose ONLY from this list: [hallway, dining room, living room, kitchen, bedroom, bathroom, entryway, storage room, gym, home office, laundry room, garage, porch].
    If no room is mentioned in the question, make a guess as to which rooms are relevant. Provide both lists as one VALID JSON dict and output nothing else!

    For example: 
    Question: What color is the kettle on the counter in the kitchen?
    Output:
    {"visual_goals":["kettle on the counter"], "rooms":["kitchen"]}

    Question: Where did I leave my blue water bottle? I can't find it.
    Output:
    {"visual_goals":["blue water bottle"], "rooms":["kitchen", "living room", "dining room", "bedroom"]}

    Question: What is on the black nightstand?
    Output:
    {"visual_goals":["object on black nightstand"] "rooms":"bedroom"]

    Question: What is the white object next to the potted plant on the table?
    Output:
    {"visual_goals":["white object next to potted plant on table"], "rooms":["living room", "dining room", "bedroom", "kitchen"]}

    Question: Is the living room coffee table the same color as the dining table?
    Output:
    {"visual_goals":["living room coffee table", "dining table"], "rooms":["living room", "dining room"]}

    Question: 
    '''

        prompt += question
        prompt += "\nOutput:"
        
        tries = 3
        while tries > 0:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
            )

            ret = response.choices[0].message.content
            logging.info(f"gpt ret: {ret}")
            parsed  = self.parse_goals(ret) #plan_json(ret)

            if parsed == None:
                tries -= 1
            else:
                return parsed
        
        return parsed

    def score_answer(self, question, answer, prediction):
        prompt = f'''You are an AI assistant who will help me to evaluate the response given the question and the correct answer.
    To mark a response, you should output a single integer between 1 and 5 (including 1, 5).
    5 means that the response perfectly matches the answer or any of the extra answers in meaning and answers the question.
    1 means that the response is completely different from the answer in terms of what it means or doesn't answer the question. 
    You should only answer with a single number!

    Example 1:
    Question: Is it overcast?
    Answer: no
    Response: yes
    Your mark: 1

    Example 2:
    Question: What color is the couch?
    Answer: white and beige
    Response: light tan
    Your mark: 3

    Example 3:
    Question: Who is standing at the table?
    Answer: a woman wearing a blue dress
    Response: a woman
    Your mark: 4

    Example 4:
    Question: what color are the curtains in the bedroom?
    Answer: the curtains are a dark blue color
    Response: they are blue
    Your mark: 5

    Your Turn:
    Question: {question}
    Answer: {answer}
    Response: {prediction}
    '''
        tries = 3
        while tries > 0:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            )

            ret = response.choices[0].message.content
            logging.info(f"gpt ret: {ret}")

            parsed = self.parse_score(ret)
            if parsed:
                return parsed
            else:
                tries -= 1
            
        return parsed
    
    def score_answer_multi(self, question, answer, extra_answers, prediction):
        prompt = f'''You are an AI assistant who will help me to evaluate the response given the question, the correct answer, and extra answers that are also correct.
    To mark a response, you should output a single integer between 1 and 5 (including 1, 5).
    5 means that the response perfectly matches the answer or any of the extra answers.
    1 means that the response is completely different from the answer and all of the extra answers.

    Example 1:
    Question: Is it overcast?
    Answer: no
    Extra Answers: ['doesn't look like it', 'no',' it's sunny']
    Response: yes
    Your mark: 1

    Example 2:
    Question: Who is standing at the table?
    Answer: woman
    Extra Answers: ['a woman', 'a lady', 'woman']
    Response: Jessica
    Your mark: 3

    Example 3:
    Question: Are there drapes to the right of the bed?
    Answer: yes
    Extra Answers: ['yes, there are drapes', 'yeah', 'the drapes are to the right of the king bed']
    Response: yes
    Your mark: 5

    Your Turn:
    Question: {question}
    Answer: {answer}
    Extra Answers: {extra_answers}
    Response: {prediction}'''
        
        tries = 3
        while tries > 0:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            )

            ret = response.choices[0].message.content
            logging.info(f"gpt ret: {ret}")

            parsed = self.parse_score(ret)
            if parsed:
                return parsed
            else:
                tries -= 1
            
        return parsed