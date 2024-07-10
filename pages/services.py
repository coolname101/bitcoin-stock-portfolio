import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title='Bitcoin Analysis', 
                   page_icon='docs/marketwise_logo.jpg', 
                   layout='wide', 
                   menu_items={'About': 'https://linkedin.com/in/itumeleng-kesebonye/'})

nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:
    st.page_link("setup.py", label="Home", icon="🏠")

with nav2:
    st.page_link("pages/services.py", label="Services", icon="1️⃣")

with nav3:
    st.page_link("pages/info.py", label="Information", icon="2️⃣")

with nav4:
    st.page_link("http://www.google.com", label="Google", icon="🌎")