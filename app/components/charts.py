import pandas as pd
import streamlit as st

def render_input_summary(values, feature_names):
    frame = pd.DataFrame({
        "Feature": feature_names,
        "Value": values,
    })
    st.dataframe(frame, use_container_width=True, hide_index=True)
