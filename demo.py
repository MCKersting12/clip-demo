from PIL import Image
import streamlit as st
from transformers import CLIPProcessor, CLIPModel
import io
import torch

try:
    model = CLIPModel.from_pretrained("/mnt/clip-vit-base-patch32-model")
    processor = CLIPProcessor.from_pretrained("/mnt/clip-vit-base-patch32-processor")

except:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.save_pretrained("/mnt/clip-vit-base-patch32-model")
    processor.save_pretrained("/mnt/clip-vit-base-patch32-processor")

st.title('CLIP model demo')

text = st.text_area('Categories (separated by commas) to sort images into', 'A photo of a dog, A photo of a cat, A photo of a person, A photo of furniture')

input_texts = [x.strip() for x in text.split(',')]
st.markdown(input_texts)

uploaded_files = st.file_uploader("Upload images for sorting",type=['jpg','jpeg','png'],help="Upload multiple images to be sorted in the format jpg,jpeg,png", accept_multiple_files=True,)

for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    image = Image.open(io.BytesIO(bytes_data))

    inputs = processor(text=input_texts, images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)[0]
    detected_index = torch.argmax(probs)
    prob = torch.max(probs).item()

    st.write("filename:", uploaded_file.name)
    st.write("Category: ", input_texts[detected_index], " - Confidence: ", prob)

    st.image(image)


