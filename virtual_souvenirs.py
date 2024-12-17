import os
import streamlit as st

def show_souvenirs():
    st.subheader("Your Virtual Souvenirs from Lakshadweep")
    st.write("Collect badges, images, and downloadable keepsakes from your travels!\n")

    # List of Lakshadweep souvenirs with captions (user provides image file path)
    souvenirs = [
        {"image_path": "Souvenirs/Beach.jpg", "caption": "Lakshadweep Beach Badge"},
        {"image_path": "Souvenirs/Coral Reef.jpg", "caption": "Lakshadweep Coral Reef Badge"},
        {"image_path": "Souvenirs/Local Tour.jpg", "caption": "Lakshadweep Local Tour Badge"},
        {"image_path": "Souvenirs/Snorkelling.jpg", "caption": "Lakshadweep Snorkeling Badge"},
        {"image_path": "Souvenirs/Scuba Diving.png", "caption": "Lakshadweep Scuba Diving Badge"},
        {"image_path": "Souvenirs/Dolphin.jpg", "caption": "Lakshadweep Dolphin Watching Badge"},
        {"image_path": "Souvenirs/Sunset Cruise.jpg", "caption": "Lakshadweep Sunset Cruise Badge"},
        {"image_path": "Souvenirs/Luxury Resort.jpg", "caption": "Lakshadweep Luxury Resort Badge"},
        {"image_path": "Souvenirs/Island Visit.jpg", "caption": "Lakshadweep Island Visit Badge"},
        {"image_path": "Souvenirs/Nature Walk.jpg", "caption": "Lakshadweep Nature Walk Badge"},
    ]

    # Display each souvenir with its image path and caption
    for souvenir in souvenirs:
        if os.path.isfile(souvenir["image_path"]):
            st.image(souvenir["image_path"], caption=souvenir["caption"], use_column_width=True)
            st.write(f"Souvenir: {souvenir['caption']}")
            st.write(f"Image File Path: {souvenir['image_path']}")
            st.write(f"To view the image, open the file at this location.\n")
        else:
            st.write(f"Souvenir: {souvenir['caption']}")
            st.write(f"Error: The image file at '{souvenir['image_path']}' could not be found.\n")
