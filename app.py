import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

# Set page configuration for a professional look
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

# Apply Custom CSS for a modern look
st.markdown("""
    <style>
        /* Background Color */
        .main { background-color: #f8f9fa; }

        /* Custom Font */
        html, body, [class*="css"] {
            font-family: 'Arial', sans-serif;
        }

        /* Stylish Buttons */
        .stButton > button {
            background-color: #007bff;
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding: 10px 24px;
            border-radius: 8px;
            border: none;
        }
        
        .stButton > button:hover {
            background-color: #0056b3;
        }

        /* Result Box */
        .result-box {
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            color: #007bff;
        }

        /* Title & Header */
        .title {
            text-align: center;
            font-size: 30px;
            font-weight: bold;
            color: #333;
        }

        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #666;
        }

        /* Align Image */
        .center {
            display: flex;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown("<div class='title'>💻 Laptop Price Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Select your laptop specifications to get an estimated price.</div>", unsafe_allow_html=True)

# Display a laptop image
st.markdown("<div class='center'><img src='https://cdn-icons-png.flaticon.com/512/1792/1792635.png' width=120></div>", unsafe_allow_html=True)

# 🖥 User Inputs
st.markdown("### 📌 Laptop Specifications")

col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("🏢 Brand", ["Apple", "HP", "Dell", "Lenovo", "Asus", "Acer", "MSI", "Toshiba", "Razer"])
    type_name = st.selectbox("💻 Laptop Type", ["Ultrabook", "Notebook", "Gaming", "2 in 1 Convertible"])
    ram = st.slider("🖥 RAM (GB)", 2, 64, step=2)
    weight = st.number_input("⚖️ Weight (kg)", min_value=0.5, max_value=5.0, step=0.1)
    ppi = st.number_input("📏 Pixels Per Inch (PPI)", min_value=50, max_value=500)

with col2:
    touchscreen = st.radio("📱 Touchscreen", ["No", "Yes"])
    ips = st.radio("📺 IPS Display", ["No", "Yes"])
    cpu_brand = st.selectbox("🔧 CPU", ["Intel Core i3", "Intel Core i5", "Intel Core i7", "Other Intel", "AMD Processor"])
    hdd = st.number_input("💾 HDD (GB)", min_value=0, max_value=2000, step=128)
    ssd = st.number_input("⚡ SSD (GB)", min_value=0, max_value=2000, step=128)
    gpu_brand = st.selectbox("🎮 GPU", ["Intel", "AMD", "Nvidia"])
    os = st.selectbox("🖥 Operating System", ["Windows", "Mac", "Other"])

# Convert categorical values to match training data
touchscreen = 1 if touchscreen == "Yes" else 0
ips = 1 if ips == "Yes" else 0

# Ensure input data has all features in correct order
input_data = np.array([[company, type_name, ram, weight, touchscreen, ips, ppi, cpu_brand, hdd, ssd, gpu_brand, os]])

# 🎯 Prediction Button
if st.button("🔮 Predict Laptop Price"):
    with st.spinner("🔄 Analyzing specs & predicting price..."):
        predicted_price = np.exp(model.predict(input_data))  # Convert back from log scale

    # 🎉 Display Result in a Card
    st.markdown(f"""
        <div class="result-box">
            💰 Estimated Price: ₹{predicted_price[0]:,.2f}
        </div>
    """, unsafe_allow_html=True)

