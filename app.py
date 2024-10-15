import streamlit as st
import sys
from pathlib import Path
PARENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0,str(PARENT_DIR))
from Insights import Overview

# Custom CSS to increase the width of the entire page
st.markdown(
    """
    <style>
    /* Adjust the width of the main content area */
    .st-emotion-cache-13ln4jf {
        max-width: 1900px; 
        margin: 0 auto;    /* Center the content */
        padding: 10px;      /* Optional padding */
    }
    
    .st-emotion-cache-12fmjuu {
    height: 0
    }

    </style>
    """,
    unsafe_allow_html=True
)
#Load the page
if __name__ == "__main__":
    data_loader= Overview()
    data_loader.load_page()
