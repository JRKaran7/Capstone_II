import streamlit as st

def group_planning():
    st.subheader("Group Travel Planning")
    st.text_input("Create a group name")
    st.text_area("Add notes for group planning")
    st.write("Invite friends by sharing this link!")
    st.button("Finalize Itinerary")
