import streamlit as st
import pandas as pd
import os

st.title('Fight Events Review')
csv_path = st.text_input('CSV path', 'outputs/fight_events.csv')
video_path = st.text_input('Video path', 'videos/fight_2.mp4')

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.dataframe(df)
else:
    st.warning('CSV not found')

st.info('Gợi ý: nếu bạn log thêm frame_idx/time_ms, có thể làm seek video tới vùng ±5s.')
