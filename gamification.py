import streamlit as st

# Simulated user rewards and badges
user_data = {
    "points": 1200,
    "badges": ["🏅 Explorer", "🎖️ Landmark Expert"],
}

def show_rewards():
    st.subheader("Your Rewards")
    st.metric("Points", user_data["points"])
    st.write(f"Badges: {', '.join(user_data['badges'])}")
