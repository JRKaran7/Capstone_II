import streamlit as st

def show_souvenirs():
    st.subheader("Your Virtual Souvenirs")
    st.write("Collect badges, images, and downloadable keepsakes from your travels!")
    st.image("https://via.placeholder.com/150", caption="Eiffel Tower Badge", width=150)
    st.image("https://via.placeholder.com/150", caption="Mount Fuji Badge", width=150)
