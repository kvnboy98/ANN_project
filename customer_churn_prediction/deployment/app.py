import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go


def set_bg_hack_url():   
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(https://t3.ftcdn.net/jpg/02/44/17/82/240_F_244178265_NP4S8WdlZRGYVSkVkxhtiDonSfQPAbyO.jpg), url(https://w0.peakpx.com/wallpaper/254/792/HD-wallpaper-blue-lego-texture-lego-background-lego-texture-blue-lego-background-constructor-texture.jpg);
             background-repeat: no-repeat;
             background-size: 800px 800px, cover;
             background-position: center;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

set_bg_hack_url()

save_model = load_model('model.hdf5')

st.markdown("<h1 style='text-align: center; color: black;'> ➡️Lego Part Classifier⬅️ </h1>", unsafe_allow_html=True)

def load_image(image_file):
	img = Image.open(image_file)
	return img

#st.subheader("Image")
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"]);

dict = {0:'10247', 1:'11090', 2:'11211', 3:'11212', 4:'11214', 5:'11458', 
     6:'11476', 7:'11477', 8:'14704', 9:'14719', 10:'14769', 11:'15068',
     12:'15070', 13:'15100', 14:'15379', 15:'15392', 16:'15535', 17:'15573',
     18:'15712', 19:'18651'}

if image_file is not None:
     col1,col2,col3 = st.columns([0.5,1,0.1])
     col2.image(load_image(image_file),width=250);
     img = load_image(image_file).resize((64,64))
     img = np.asarray(img)
     img = img.reshape((1,64,64,3))
     img = img/255.0

st.markdown("<h5 style='text-align: center; color: black;'> Click bellow to start predict </h5>", unsafe_allow_html=True)
col16, col27, col38 = st.columns([1.5, 1, 1])
pred_df = pd.DataFrame({'prediction': [0]})
prob_df = pd.DataFrame({'prob': np.zeros([1,20])[0].tolist(), 
                         'part': dict.values()})

if col27.button('Predict'):
     y_preds_prob = save_model.predict(img)
     
     prob_df['prob'] = y_preds_prob.tolist()[0]
     prob_df = prob_df.reset_index().sort_values('prob', ascending=True)
     prob_df['part'] = prob_df['index'].map(dict)
     #prob_df

     y_preds = np.argmax(y_preds_prob)
     pred_df['prediction'] = y_preds

     pred_df['prediction'] = pred_df['prediction'].map(dict)

     fig = go.Figure(go.Bar(x=prob_df['prob'][-5:],
                    y=prob_df['part'].iloc[-5:],
                    orientation='h'))
     fig.update_layout(autosize=False, margin={'l':20, 'r':20, 't':20, 'b':20},
          width=680,
          height=400, paper_bgcolor='rgb(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
          )
     st.markdown(f"<h1 style='text-align: center; color: black;'> {pred_df['prediction'].iloc[0]} </h1>", unsafe_allow_html=True)
     if prob_df['prob'].iloc[19] <= 0.8:
          st.markdown("<h5 style='text-align: center; color: red;'> Confidence level low, guess might wrong! </h5>", unsafe_allow_html=True)
     with st.expander("See Confidence"):
          st.plotly_chart(fig)
          


else:
     pred_df['prediction'].iloc[0] = '-'
     st.markdown(f"<h1 style='text-align: center; color: black;'> {pred_df['prediction'].iloc[0]} </h1>", unsafe_allow_html=True)