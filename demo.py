# pip install git+https://github.com/


import torch
import copy
from PIL import Image
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images

pretrained = "zhibinlan/LLaVE-0.5B"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()

# Image + Text -> Text
image = Image.open("figures/example.jpg")
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

question = DEFAULT_IMAGE_TOKEN + " Represent the given image with the following question: What is in the image"
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], "\n")
prompt_question = conv.get_prompt()
input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
attention_mask=input_ids.ne(tokenizer.pad_token_id)
image_sizes = [image.size]
query_embed = model.encode_multimodal_embeddings(input_ids, attention_mask=attention_mask,images=image_tensor, image_sizes=image_sizes)

target_string = "A cat and a dog"
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], target_string)
conv.append_message(conv.roles[1], "\n")
target_string = conv.get_prompt()
target_input_ids = tokenizer(target_string, return_tensors="pt").input_ids.to(device)
attention_mask=target_input_ids.ne(tokenizer.pad_token_id)
target_embed = model.encode_multimodal_embeddings(target_input_ids, attention_mask=attention_mask)

print("A cat and a dog similarity score: ", query_embed @ target_embed.T)
# 0.5B: A cat and a dog similarity score: tensor([[0.4802]]
# 2B: A cat and a dog similarity score: tensor([[0.5132]]
# 7B: A cat and a dog similarity score: tensor([[0.6240]]

neg_string = "A cat and a tiger"
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], neg_string)
conv.append_message(conv.roles[1], "\n")
neg_string = conv.get_prompt()
neg_input_ids = tokenizer(neg_string, return_tensors="pt").input_ids.to(device)
attention_mask=neg_input_ids.ne(tokenizer.pad_token_id)
neg_embed = model.encode_multimodal_embeddings(neg_input_ids, attention_mask=attention_mask)
print("A cat and a tiger similarity score: ", query_embed @ neg_embed.T)
# 0.5B: A cat and a tiger similarity score: tensor([[0.3413]]
# 2B: A cat and a tiger similarity score: tensor([[0.3809]]
# 7B: A cat and a tiger similarity score: tensor([[0.4543]]


# Text -> Image
pos_string = "Find me an everyday image that matches the given caption: A cat and a dog."
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], pos_string)
conv.append_message(conv.roles[1], "\n")
pos_string = conv.get_prompt()
pos_input_ids = tokenizer(pos_string, return_tensors="pt").input_ids.to(device)
attention_mask=pos_input_ids.ne(tokenizer.pad_token_id)
pos_query_embed = model.encode_multimodal_embeddings(pos_input_ids, attention_mask=attention_mask)

target = DEFAULT_IMAGE_TOKEN + " Represent the given image."
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], target)
conv.append_message(conv.roles[1], "\n")
prompt_target = conv.get_prompt()
target_input_ids = tokenizer_image_token(prompt_target, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
attention_mask=target_input_ids.ne(tokenizer.pad_token_id)
target_image_sizes = [image.size]
target_embed = model.encode_multimodal_embeddings(target_input_ids, attention_mask=attention_mask,images=image_tensor, image_sizes=target_image_sizes)

print("A cat and a dog image similarity score: ", pos_query_embed @ target_embed.T)
# 0.5B: A cat and a dog similarity score: tensor([[0.3992]]
# 2B: A cat and a dog similarity score: tensor([[0.5225]]
# 7B: A cat and a dog similarity score: tensor([[0.5347]]

neg_string = "Find me an everyday image that matches the given caption: A cat and a tiger."
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], neg_string)
conv.append_message(conv.roles[1], "\n")
neg_string = conv.get_prompt()
neg_input_ids = tokenizer(neg_string, return_tensors="pt").input_ids.to(device)
attention_mask=neg_input_ids.ne(tokenizer.pad_token_id)
neg_query_embed = model.encode_multimodal_embeddings(neg_input_ids, attention_mask=attention_mask)

print("A cat and a tiger image similarity score: ", neg_query_embed @ target_embed.T)
# 0.5B: A cat and a dog similarity score: tensor([[0.3354]]
# 2B: A cat and a dog similarity score: tensor([[0.4141]]
# 7B: A cat and a dog similarity score: tensor([[0.4001]]