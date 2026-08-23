import streamlit as st
from diffusers import DiffusionPipeline
import torch

st.title("🎨 AI Image Generator")

prompt = st.text_input(
    "Enter your prompt:",
    "A cute cat sitting in a garden"
)

if st.button("Generate Image"):
    with st.spinner("Generating image..."):

        pipe = DiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            torch_dtype=torch.float32
        )

        image = pipe(prompt).images[0]

        st.image(
            image,
            caption="Generated Image"
        )
