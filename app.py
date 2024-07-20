import  streamlit as st
import pandas as pd
import numpy as np
import pickle
st.title('Mushroom Classification')
st.write('This is a simple web app to predict the whether the mushroom is edible or poisonous')
st.write('Please input the following information:')
# Load the model
model = pickle.load(open('model.pkl', 'rb'))
# Take the input from the user
col1,col2,col3=st.columns(3)
with col1:
    cap_diameter=st.number_input('mushroom cap diameter',min_value=0,max_value=2000,step=1,value=0)
with col2:
    cap_shape=st.slider('mushroom cap shape',min_value=0,max_value=6,step=1)
with col3:
    gill_attachment=st.slider('gill attachment',min_value=0,max_value=12,step=1)
col4,col5,col6=st.columns(3)
with col4:
    gill_color=st.slider('gill color',min_value=0,max_value=12,step=1)
with col5:
    stem_height=st.slider('stem height',min_value=0.0,max_value=4.0,step=0.1)
with col6:
    stem_width=st.number_input('stem width',min_value=0,max_value=4000,step=1,value=0)
col7,col8=st.columns(2)
with col7:
    stem_color=st.slider('stem color',min_value=0,max_value=12,step=1)
with col8:
    season=st.slider('season',min_value=0.0,max_value=2.0,step=0.1)
# Create a DataFrame for the model
input_data = {'cap-diameter': cap_diameter, 'cap-shape': cap_shape, 'gill-attachment': gill_attachment, 'gill-color': gill_color, 'stem-height': stem_height, 'stem-width': stem_width, 'stem-color': stem_color, 'season': season}
input_df = pd.DataFrame([input_data])
# Predict the output
if st.button('Predict'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    st.subheader('The probability of mushroom being edible is:'+ str(round(prediction_proba[0][0],2)) +'%')
    if prediction[0] == 0:
        st.success('The mushroom is edible')
    else:
        st.error('The mushroom is poisonous')




