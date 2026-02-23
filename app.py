import streamlit as st, pandas as pd, joblib

st.title("Credit Score Predictor")
# 1. Load Artifacts
try:
    data = joblib.load('model.pkl')
    model, encoders, cols = data['model'], data['encoders'], data['cols']
except:
    st.error("Run 'python train.py' first!"); st.stop()

# 2. Sidebar Inputs
st.sidebar.header("User Data")
inputs = {}
for c in cols:
    if c in encoders:
        inputs[c] = st.sidebar.selectbox(f"Select {c}", encoders[c].classes_)
    else:
        inputs[c] = st.sidebar.number_input(f"Enter {c}", value=0.0)

# 3. Predict Result
if st.button("Predict Score"):
    df = pd.DataFrame([inputs])
    for c, le in encoders.items():
        if c in df.columns: df[c] = le.transform(df[c].astype(str))
    res = model.predict(df)[0]
    if 'Credit_Score' in encoders: res = encoders['Credit_Score'].inverse_transform([res])[0]
    st.success(f"Result: {res}"); st.balloons()
