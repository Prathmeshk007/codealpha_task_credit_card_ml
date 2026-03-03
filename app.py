import streamlit as st
import pandas as pd
import pickle


with open('model.pkl', 'rb') as f:
    saved_data = pickle.load(f)
model = saved_data['model']
features = saved_data['features']



st.set_page_config(page_title="Credit Predictor", page_icon="💳")
st.title("💳 Simple Credit Score Predictor")
st.write("Fill in the details below to see your predicted credit score.")

left_col, right_col = st.columns(2)

with left_col:
    income = st.number_input("Annual Income ($)", value=50000)
    age = st.number_input("Age", value=25)
    banks = st.number_input("Bank Accounts", value=2)

with right_col:
    cards = st.number_input("Number of Credit Cards", value=3)
    rate = st.slider("Interest Rate (%)", 0, 40, 15)


if st.button("Predict Score"):
    user_input = pd.DataFrame([[income, age, banks, cards, rate]], columns=features)
    
    result = model.predict(user_input)[0]
    
    st.divider()
    if result == 'Good':
        st.success(f"### Predicted Score: **{result}** 🎉")
    elif result == 'Standard':
        st.info(f"### Predicted Score: **{result}** 👍")
    else:
        st.error(f"### Predicted Score: **{result}** ⚠️")
    st.balloons()
