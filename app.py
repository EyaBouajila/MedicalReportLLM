import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv('google_api')
genai.configure(api_key=GOOGLE_API_KEY)

# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, convert_system_message_to_human=True, google_api_key=os.getenv("google_api"))
generation_config = {
    "temperature" : 0.4,
    "top_p" : 1,
    "top_k" : 32,
    "max_output_tokens" : 4096,
}

model = genai.GenerativeModel(model_name="gemini-1.5-flash",generation_config=generation_config)

st.set_page_config(page_title="Medical Image Analytics", page_icon=":robot:")
st.title("Medical Image Analytics 🩺🥼💉")
st.subheader("An App that will help users identify medical images")

uploaded_file = st.file_uploader("Upload the medical image for analysis", type=["png","jpg","jpeg"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Medical Image")

submit_button = st.button("Generate the Analysis")

system_prompt = """
    As a highly skilled medical practitioner specializing in image analysis, you are tasked with examining medical images for a renowned hospital. 
    your expertise is crucial in identifying any anomalies, diseases, or health issues that may be present in the images.
    your responsibilities include :
    1. detailed analysis: thoroughly analyze each image, focusing on identifying any abnormal findings.
    2. findings report: document all observed anomalies or signs of disease. clearly articulate these findings ina   structured format.
    3. recommendations and next steps : based on your analysis,suggest potential next steps, including further tests or treatments as applicable.
    4. treatment suggestions: if appropriate, recommend possible treatment options or interventions.
    
    important notes :
    1. scope of response : only respond if the image pertains to human health issues.
    2. clarity of image : in cases where the image quality impedes clear analysis, note that certain aspects are 'unable to be determined based on the provided image'
    3. disclaimer : accompany your analysis with the disclaimer : 'consult with a doctor before making any decisions'
    4. your insights are invaluable in guiding clinical decisions. please proceed with the analysis, adhering to the structured approach outlined above
    """

if submit_button:
    image_data=uploaded_file.getvalue()
    image_parts = [
        {
            "mime_type":"image/jpeg",
            "data":image_data
        }
    ]
    prompt_parts = [
        image_parts[0],
        system_prompt
    ]
    st.title("Here is the analysis based on your uploaded image :")
    response = model.generate_content(prompt_parts)
    # print(response.text)
    st.write(response.text)