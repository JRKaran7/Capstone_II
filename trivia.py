import streamlit as st
import random
import json

def load_trivia():
    with open("trivia_questions.json", "r") as file:
        return json.load(file)

def show_trivia():
    st.subheader("Test Your Travel Knowledge!")
    trivia_data = load_trivia()
    question = random.choice(trivia_data)
    
    st.write(f"**Question:** {question['question']}")
    for i, option in enumerate(question["options"], start=1):
        if st.button(f"{i}. {option}"):
            if option == question["answer"]:
                st.success("Correct! 🎉")
            else:
                st.error(f"Wrong! The correct answer is: {question['answer']}")
                break
