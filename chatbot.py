import streamlit as st

def start_chatbot():
    st.subheader("Travel Chatbot")
    query = st.text_input("Ask me anything about travel!")
    if query:
        st.write("🤖 Processing your query...")
        st.write(f"**Response:** Here's a suggestion for '{query}'.")
