import streamlit as st

def render_prediction(result):
    st.success(f"Predicted quality: {result['quality']}")

    st.subheader("Class probabilities")

    for label, probability in result["probabilities"].items():
        st.write(f"{label}: {probability:.2%}")
        st.progress(probability)
