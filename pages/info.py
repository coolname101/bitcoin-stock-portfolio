import streamlit as st

st.set_page_config(page_title='Bitcoin Analysis', 
                   page_icon='docs/marketwise_logo.jpg',
                   initial_sidebar_state='collapsed', 
                   layout='wide', 
                   menu_items={'About': 'https://linkedin.com/in/itumeleng-kesebonye/'})

background = """
<style>
    .stApp {
        background: linear-gradient(-45deg, #F3F8FF, #F5F5F5, #FCFAEE, #EEF7FF);
	    background-size: 400% 400%;
	    animation: gradient 15s ease infinite;
	    height: 100vh;
    }
    
    [data-testid="collapsedControl"] {
        display: none;
    }
    
    .st-emotion-cache-79elbk {
        background: rgba(0,0,0,0));
     }
    
    st-emotion-cache-1mi2ry5 {
        background-color: #FCFAEE;
    }
    
    .st-emotion-cache-1l269bu {
        background-color: rgba(0,0,0,0);
    }
    
    @keyframes gradient {
	0% {
		background-position: 0% 50%;
	}
	50% {
		background-position: 100% 50%;
	}
	100% {
		background-position: 0% 50%;
	}
}

</style>
"""

st.markdown(background, unsafe_allow_html=True)

nav1, nav2, nav3, nav4 = st.columns([0.115, 0.115, 0.115, 0.6])

with nav1:
    st.page_link("app.py", label="BTC Dashboard", icon="🏠")

with nav2:
    st.page_link("pages/services.py", label="Stock Dashboard", icon="1️⃣")

with nav3:
    st.page_link("pages/info.py", label="Information", icon="2️⃣")
    
image, title = st.columns([0.25, 3])

with image:
    st.image('docs/marketwise_logo.jpg', 
         caption='', width=80)
    
with title:
    st.title('Information')

page_info = ['Bitcoin Dashboard', 'Stock Dashboard']

options = st.selectbox("Please select dashboard", page_info)

