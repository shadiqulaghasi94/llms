import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained BLIP preprocessor and model
preprocessor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def give_a_caption(input_image: np.ndarray):
    # Convert numpy array to PIL Image and convert to RGB
    raw_image = Image.fromarray(input_image).convert('RGB')

    # Ensure the input image is in the correct format
    if not isinstance(input_image, np.ndarray):
        raise ValueError("Input image must be a numpy array.")
    
    # Process the image
    inputs = preprocessor(raw_image, return_tensors="pt")

    # Generate a caption for the image
    out = model.generate(**inputs,max_length=50)

    # Decode the generated tokens to text
    caption = preprocessor.decode(out[0], skip_special_tokens=True)

    return caption


gr_interface = gr.Interface(
    fn=give_a_caption, 
    inputs=gr.Image(), 
    outputs="text",
    title="Image Captioning Tool",
    description="This web app generates captions for images using the BLIP model."
)

gr_interface.launch(debug=True, share=False)