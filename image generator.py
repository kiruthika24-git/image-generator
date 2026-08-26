import streamlit as st
from diffusers import DiffusionPipeline

st.title("🎨 AI Image Generator")

prompt = st.text_input(
    "Enter your prompt:",
    "A cute cat sitting in a beautiful garden"
)

if st.button("Generate Image"):

    with st.spinner("Generating image..."):

        pipe = DiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5"
        )

        image = pipe(prompt).images[0]

        st.image(
            image,
            caption="Generated Image"
        )
