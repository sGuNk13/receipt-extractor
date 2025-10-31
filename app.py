import streamlit as st
from PIL import Image
import json
import os
from extractor import extract_receipt_data

st.set_page_config(page_title="Receipt Extractor", layout="wide")

st.title("Receipt Data Extractor")

# Get API key from secrets or input
groq_api_key = st.secrets.get("GROQ_API_KEY", "")
if not groq_api_key:
    groq_api_key = st.text_input("Enter Groq API Key:", type="password")

if groq_api_key:
    uploaded_file = st.file_uploader("Upload receipt (image or PDF)", type=['jpg', 'jpeg', 'png', 'pdf'])
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Receipt Image")
            if uploaded_file.type != "application/pdf":
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
            else:
                st.info("PDF preview not available")
        
        with col2:
            st.subheader("Extracted Data")
            if st.button("Extract Data"):
                with st.spinner("Extracting..."):
                    # Save temp file
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    try:
                        result = extract_receipt_data(temp_path, groq_api_key)
                        st.json(result)
                        
                        # Download button
                        st.download_button(
                            "Download JSON",
                            data=json.dumps(result, indent=2),
                            file_name="receipt_data.json",
                            mime="application/json"
                        )
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                    finally:
                        os.remove(temp_path)
else:
    st.warning("Please enter your Groq API key to continue")
