from pyexpat import features
import streamlit as st
from datetime import datetime
import requests
from ml_model import predict_crop
import pandas as pd
import numpy as np
import joblib
import altair as alt


from disease_model import predict_disease

# --- Load saved objects and define prediction function ---
model = joblib.load("crop_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
model_features = joblib.load("model_features.pkl")  # if you saved this

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    # Feature engineering
    NPK = (N + P + K) / 3
    THI = temperature * humidity / 100
    temp_rain_interaction = temperature * rainfall
    ph_rain_interaction = ph * rainfall

    # Encode categorical features
    rainfall_level = 0 if rainfall <= 50 else 1 if rainfall <= 100 else 2 if rainfall <= 200 else 3
    ph_category = 0 if ph < 5.5 else 1 if ph <= 7.5 else 2

    # Make DataFrame in correct order
    features = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall,
                              NPK, THI, rainfall_level, ph_category,
                              temp_rain_interaction, ph_rain_interaction]],
                            columns=model_features)

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict probabilities
    probs = model.predict_proba(features_scaled)[0]

    # Top 3 crops
    top3_idx = probs.argsort()[-3:][::-1]
    top3_crops = label_encoder.inverse_transform(top3_idx)

    return top3_crops


# --- Streamlit pages start below ---

def engineer_features(N, P, K, temperature, humidity, ph, rainfall):
    # Create engineered features
    NPK = (N + P + K) / 3
    THI = temperature * humidity / 100
    temp_rain_interaction = temperature * rainfall
    ph_rain_interaction = ph * rainfall

    # Categorize rainfall
    if rainfall <= 50:
        rainfall_level = 0  # Low
    elif rainfall <= 100:
        rainfall_level = 1  # Medium
    elif rainfall <= 200:
        rainfall_level = 2  # High
    else:
        rainfall_level = 3  # Very High

    # Categorize pH
    if ph < 5.5:
        ph_category = 0  # Acidic
    elif ph <= 7.5:
        ph_category = 1  # Neutral
    else:
        ph_category = 2  # Alkaline

    # Create dataframe with correct feature order (same as model)
    features = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall, NPK, THI,
                              rainfall_level, ph_category, temp_rain_interaction, ph_rain_interaction]],
                            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall',
                                     'NPK', 'THI', 'rainfall_level', 'ph_category',
                                     'temp_rain_interaction', 'ph_rain_interaction'])
    return features


# Page config (only call once)
st.set_page_config(page_title="KISAN Sahayak", page_icon="🌱", layout="wide")


st.markdown("""
<style>
/* --- App Background & Text --- */
.stApp {
    background-color: #e6f4ea; /* light green */
    color: #1b5e20; /* dark green text */
}

/* --- Sidebar --- */
section[data-testid="stSidebar"] {
    background-color: #2e7d32;
}
section[data-testid="stSidebar"] * {
    color: white;
}

/* --- Headers --- */
h1, h2, h3, h4, h5, h6 {
    color: #1b5e20;
}

/* --- Buttons --- */
div.stButton > button {
    background-color: #388e3c;
    color: white;
    border-radius: 8px;
    font-weight: 600;
}
div.stButton > button:hover {
    background-color: #4caf50;
}

/* --- Inputs / Sliders / Selectboxes --- */
div.stTextInput > div > input,
div.stSlider > div > input,
div.stSelectbox > div > select {
    border-radius: 8px;
    border: 1px solid #388e3c;
    background-color: #f1f8f2;
    color: #1b5e20;
    font-weight: 500;
}

/* Labels */
.stSlider label,
.stSelectbox label,
.stTextInput label {
    color: #1a3d1a !important;
    font-weight: 600;
}

/* --- Success Messages --- */
div.stAlert.stAlert-success {
    background-color: #ffffff !important; /* white background */
    border: 2px solid #155724;            /* dark green border */
    border-radius: 6px;
}

div.stAlert.stAlert-success * {
    color: #155724 !important;  /* dark green text */
    font-weight: 700;
    font-size: 18px;
}

/* --- Error Messages --- */
div.stAlert.stAlert-error,
div.stAlert.stAlert-error * {
    color: #721c24 !important;
    background-color: #f8d7da !important;
    font-weight: 700;
    font-size: 18px;
}

/* --- Info Messages --- */
div.stAlert.stAlert-info,
div.stAlert.stAlert-info * {
    color: #0c5460 !important;
    background-color: #d1ecf1 !important;
    font-weight: 700;
    font-size: 16px;
}
            
/* --- Force All Main Text to Dark Green --- */
.stApp, .stApp div, .stApp span, .stApp p {
    color: #1b5e20 !important; /* dark green */
    font-weight: 500;
}

/* --- White Background + Dark Green Text --- */
.stApp {
    background-color: #ffffff !important; /* white */
}
.stApp, .stApp div, .stApp span, .stApp p {
    color: #1b5e20 !important; /* dark green text */
    font-weight: 600;
}

/* --- Sidebar Fix: Force White Text --- */
section[data-testid="stSidebar"] * {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Menu")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Profile", "Predict Your Crop", "More Features"],
    key="menu_radio"
)

# --- Home Page ---
if page == "Home":
    # 🔍 Search Bar
    st.markdown("### 🔍 What are you looking for?")
    col1, col2 = st.columns([8,1])
    with col1:
        query = st.text_input(
            "Type your query here...",
            placeholder="e.g., recommend crop, rice price, mango disease",
            label_visibility="collapsed"
        )
    with col2:
        st.button("🎤", help="Click to speak your query")

    # 🌱 App Header Box
    st.markdown("""
    <div style="text-align:center; padding: 20px; background-color:#e6f4ea; border-radius:10px; border:2px solid #388e3c;">
        <h1 style="color:#1b5e20;">🌱 KISAN Sahayak</h1>
        <h3 style="color:#388e3c;">Growing Tomorrow's Food Today</h3>
    </div>
    """, unsafe_allow_html=True)

    # Tagline
    st.subheader("Welcome to KISAN Sahayak, Akash")

    # Dashboard + Weather side by side
    dash_col, weather_col = st.columns([1, 1])

    with dash_col:
        st.markdown("""
        <div style="background:#ffffff; border:2px solid #388e3c; border-radius:10px; padding:15px;">
            <h3 style="color:#1b5e20;">🌾 Your Farm</h3>
            <p><b>Soil pH:</b> 6.5 (Neutral)</p>
            <p><b>Soil Moisture:</b> Low</p>
            <p><b>Last Recommendation:</b> Maize 🌽</p>
        </div>
        """, unsafe_allow_html=True)

        
    
    with weather_col:
        now = datetime.now()
        try:
            city = "Delhi"  # Change this dynamically later
            url = f"http://wttr.in/{city}?format=j1"  # JSON format
            headers = {"User-Agent": "Mozilla/5.0"}   # Pretend it's a browser
            response = requests.get(url, headers=headers, timeout=5)

            # Debugging step (to check what’s coming back)
            if response.headers.get("Content-Type") != "application/json":
                raise ValueError("Non-JSON response received")

            data = response.json()

            # Extract current condition
            temp = data['current_condition'][0]['temp_C']
            hum = data['current_condition'][0]['humidity']
            desc = data['current_condition'][0]['weatherDesc'][0]['value']

            weather_info = f"""
            <div style="background:#f1f8f2; border:2px solid #1b5e20; border-radius:10px; padding:15px; min-height:200px;">
                <p><b>Date 📅:</b> {now.strftime('%A, %d %B %Y')}</p>
                <p><b>Current time ⏰:</b> {now.strftime('%H:%M:%S')}</p>
                <p><b>City:</b> {city}</p>
                <p><b>Temperature 🌡:</b> {temp} °C</p>
                <p><b>Humidity 💧:</b> {hum} %</p>
                <p><b>Condition ☁:</b> {desc}</p>
            </div>
            """
        except Exception as e:
            weather_info = f"""
            <div style="background:#f1f8f2; border:2px solid #1b5e20; border-radius:10px; padding:15px; min-height:200px;">
                <p><b>Date 📅:</b> {now.strftime('%A, %d %B %Y')}</p>
                <p><b>Current time ⏰:</b> {now.strftime('%H:%M:%S')}</p>
                <p>⚠ Weather data unavailable: {e}</p>
            </div>
            """

        st.markdown(weather_info, unsafe_allow_html=True)

        # 🌱 Community Banner
    st.markdown("""
    <div style="background:#dcedc8; border:2px solid #1b5e20; border-radius:10px; padding:20px; text-align:center;">
        <h2 style="color:#33691e;">👩‍🌾 Community Stories</h2>
        <p>Share your farm experiences, tips, and success stories!</p>
        <img src="https://via.placeholder.com/400x150" style="border-radius:10px; margin-top:10px;" />
    </div>
    """, unsafe_allow_html=True)




# --- Profile Page ---


elif page == "Profile":
    st.title("Your Profile")
    
    # --- Section 1: Personal Info and Farm Details 🧑‍🌾 ---
    st.subheader("Personal Information & Farm Details")
    st.image("https://via.placeholder.com/120", width=120)  # Profile picture placeholder

    # Columns for half-width inputs
    col1, col2 = st.columns([1, 1])

    with col1:
        st.text_input("Farmer's Name", value="Akash", key="name")
        st.text_input("Email", value="akash@example.com", key="email")
        st.text_input("Phone", value="+91 99999 99999", key="phone")
        st.text_input("Age", value="30", key="age")      

    with col2:
        st.text_input("Farm Name", value="Green Valley Farm", key="farm_name")
        st.text_input("Location", value="Pune, Maharashtra", key="location")
        st.text_input("Farm Size (acres)", value="5.2", key="farm_size")
        st.text_input("Soil Type", value="Clay Loam", key="soil_type")
        st.text_input("Irrigation", value="Drip Irrigation", key="irrigation")

    st.markdown("---")  # Divider

    # --- Inject CSS for Profile page ---
    st.markdown("""
    <style>
    div.stTextInput > div > input {
        width: 90% !important;                /* smaller width */
        background-color: #e6f4ea !important;  /* light green */
        border: 2px solid #388e3c !important;  /* darker green border */
        border-radius: 8px !important;
        padding: 6px 10px !important;
        color: #1b5e20 !important;
        font-weight: 500 !important;
    }
    div.stTextInput > div > input::placeholder {
        color: #2e7d32 !important;
        opacity: 0.7;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Section 2: History and Activity Log 📜 ---
    with st.expander("📜 History & Activity Log"):
        st.write("Diagnosis History")
        st.dataframe({
            "Date": ["23-Sep-2025", "20-Sep-2025"],
            "Detected Disease": ["Leaf Blight", "Powdery Mildew"],
            "Recommended Solution": ["Fungicide Spray", "Neem Oil"]
        })
        
        st.write("Profitability Reports")
        st.dataframe({
            "Crop": ["Wheat", "Tomatoes"],
            "Yield (kg)": [1200, 800],
            "Profit (₹)": [25000, 18000],
            "Sustainability Score": ["Good", "Moderate"]
        })

    st.markdown("---")  # Divider

    # --- Section 3: App Settings and Preferences ⚙️ ---
    with st.expander("⚙️ App Settings and Preferences"):
        # Language Selection
        language = st.selectbox("Select Language", ["English", "Hindi", "Marathi"])

        # Notification Settings
        st.subheader("Notifications & Alerts")
        st.checkbox("Weather Alerts", value=True)
        st.checkbox("Pest Alerts", value=True)
        st.checkbox("Market Updates", value=True)

        # Voice Assistant Settings
        st.subheader("Voice Assistant Settings")
        st.slider("Voice Assistant Volume", 0, 100, 50)
        st.selectbox("Voice / Dialect", ["Default", "Male", "Female", "Regional Accent"])

        # Account Management
        st.subheader("Account Management")
        st.text_input("Change Password", type="password")
        st.button("Log Out")



# --- Prediction Page ---
elif page == "Predict Your Crop":
    st.title("🌱 Crop Recommendation System")
    st.markdown("Provide your soil and weather details to get the best crop recommendation:")
    col1, col2 = st.columns(2)
    with col1:
        nitrogen = st.slider("Nitrogen (N)", 0, 150, 50)
        phosphorus = st.slider("Phosphorus (P)", 0, 150, 50)
        potassium = st.slider("Potassium (K)", 0, 200, 50)
        ph = st.slider("Soil pH", 0.0, 14.0, 6.5, 0.1)
    with col2:
        temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0, 0.5)
        humidity = st.slider("Humidity (%)", 0, 100, 60)
        rainfall = st.slider("Rainfall (mm)", 0, 300, 100)

    if st.button("🌾 Predict Best Crop"):
        top_crops = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
        st.success(f"✅ Recommended Crops: **{', '.join(top_crops)}**")

import streamlit as st
import pandas as pd
import numpy as np

# Placeholder predict_disease function
def predict_disease(image_file):
    return "Unknown Disease", "No recommended action yet"




# --- More Features Page ---
if page == "More Features":
    st.title("🌟 More Features")

    feature = st.radio(
        "Select Feature",
        ["About the app", "Disease Diagnosis", "Marketplace", "Get Crop Advice",
         "Ask Assistant", "Alerts & Highlights", "Plant Helpdesk"]
    )

    # ---------------- About the app ----------------
    if feature == "About the app":
        st.subheader("📚 About the app")
        st.write("Your one-stop solution for all farming needs — collaborate, learn and grow.")
        contribution = st.text_input("Enter your contribution (story/tip/guide)", key="about_contrib")
        if st.button("Show Contribution", key="btn_show_contrib"):
            if contribution.strip():
                st.success("Thanks — contribution saved (demo).")
                st.info(f"Preview: {contribution}")
            else:
                st.warning("Please enter something to share.")

    # --- Disease Diagnosis ---
    elif feature == "Disease Diagnosis":
        st.subheader("🦠 Disease Diagnosis")
        st.write("Upload an image of your plant to detect possible diseases.")
        uploaded_file = st.file_uploader("Choose a plant image...", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
            if st.button("🔍 Diagnose Disease"):
                disease, action = predict_disease(uploaded_file)
                st.success(f"Predicted Disease: {disease}")
                st.info(f"Recommended Action: {action}")

    # --- Marketplace ---
    elif feature == "Marketplace":
        st.subheader("🛒 Marketplace")
        st.write("Browse seeds, fertilizers, and other farming products.")

        # Example marketplace data
        market_data = pd.DataFrame([
            {"Crop": "Tomato", "Category": "Vegetable", "Price": 30, "Unit": "kg", "Location": "Pune", "Quality": "A",
             "Trend": [28, 29, 30, 31, 32, 30, 30], "Image": "https://via.placeholder.com/150"},
            {"Crop": "Wheat", "Category": "Cereal", "Price": 25, "Unit": "kg", "Location": "Delhi", "Quality": "B",
             "Trend": [24, 25, 26, 25, 27, 26, 25], "Image": "https://via.placeholder.com/150"},
            {"Crop": "Maize", "Category": "Cereal", "Price": 20, "Unit": "kg", "Location": "Hyderabad", "Quality": "A",
             "Trend": [18, 20, 19, 21, 22, 20, 21], "Image": "https://via.placeholder.com/150"},
            {"Crop": "Brinjal", "Category": "Vegetable", "Price": 35, "Unit": "kg", "Location": "Bangalore", "Quality": "B",
             "Trend": [32, 33, 34, 35, 36, 34, 35], "Image": "https://via.placeholder.com/150"},
        ])

        # Sidebar filters
        st.sidebar.header("Filter & Sort")
        search_crop = st.sidebar.text_input("Search Crop")
        category_filter = st.sidebar.multiselect(
            "Category",
            options=market_data["Category"].unique(),
            default=market_data["Category"].unique()
        )
        min_price, max_price = st.sidebar.slider(
            "Price Range (₹)",
            int(market_data["Price"].min()),
            int(market_data["Price"].max()),
            (int(market_data["Price"].min()), int(market_data["Price"].max()))
        )
        sort_option = st.sidebar.selectbox("Sort by", ["Price: Low → High", "Price: High → Low", "Trending"])

        # Apply filters
        filtered = market_data[
            (market_data["Crop"].str.contains(search_crop, case=False)) &
            (market_data["Category"].isin(category_filter)) &
            (market_data["Price"] >= min_price) &
            (market_data["Price"] <= max_price)
        ]

        # Sorting
        if sort_option == "Price: Low → High":
            filtered = filtered.sort_values("Price")
        elif sort_option == "Price: High → Low":
            filtered = filtered.sort_values("Price", ascending=False)
        elif sort_option == "Trending":
            filtered["TrendMean"] = filtered["Trend"].apply(lambda x: np.mean(x[-3:]) - np.mean(x[:3]))
            filtered = filtered.sort_values("TrendMean", ascending=False)

        # Inject CSS for boxes and buttons
        st.markdown("""
        <style>
        .card-box {
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 20px;
            background-color: #f9f9f9;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
        }
        .contact-btn {
            background-color: #4caf50;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 6px 12px;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)

        # Display marketplace items in 2 columns
        cols = st.columns(2)
        for idx, row in filtered.iterrows():
            with cols[idx % 2]:
                st.markdown(f'<div class="card-box">', unsafe_allow_html=True)
                st.image(row["Image"], width=180)
                st.subheader(f"{row['Crop']} ({row['Quality']})")
                st.markdown(f"**Price:** {row['Price']} {row['Unit']}")
                st.markdown(f"**Location:** {row['Location']}")
                with st.expander("View 7-day Price Trend 📈"):
                    trend_df = pd.DataFrame({"Day": range(1, 8), "Price": row["Trend"]})
                    st.line_chart(trend_df.rename(columns={"Day": "Index", "Price": "Price"}).set_index("Index"))
                st.markdown(f'<div class="contact-btn">📞 Contact Seller</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Get Crop Advice ----------------
    elif feature == "Get Crop Advice":
        st.title("🌾 Get Crop Advice")
        st.markdown("Get personalized recommendations based on your soil, weather and market inputs.")

        with st.form("crop_advice_form", clear_on_submit=False):
            col_a, col_b = st.columns(2)
            with col_a:
                n = st.number_input("Nitrogen (N)", 0, 200, 50, key="adv_n")
                p = st.number_input("Phosphorus (P)", 0, 200, 30, key="adv_p")
                k = st.number_input("Potassium (K)", 0, 200, 40, key="adv_k")
                ph_val = st.number_input("Soil pH", 3.0, 9.0, 6.5, key="adv_ph")
            with col_b:
                rainfall = st.number_input("Rainfall (mm)", 0, 500, 120, key="adv_rain")
                temp = st.slider("Temperature (°C)", 5, 45, 28, key="adv_temp")
                humidity = st.slider("Humidity (%)", 10, 100, 60, key="adv_hum")
                market_demand = st.selectbox("Market Demand", ["High", "Medium", "Low"], key="adv_demand")

            submitted = st.form_submit_button("🌱 Recommend Crops")

        if submitted:
            # Try to use your model function; fallback to mock if it fails
            try:
                recommended = predict_crop(n, p, k, temp, humidity, ph_val, rainfall)
                # predict_crop may return array-like — convert to list
                recommended = list(recommended) if hasattr(recommended, "__iter__") else [recommended]
            except Exception:
                recommended = ["Wheat", "Paddy", "Maize"]

            # Example extra info per crop (demo numbers)
            info_map = {
                "Wheat": {"yield": 2800, "profit": 34000, "water": "Medium", "sustain": "High"},
                "Paddy": {"yield": 3500, "profit": 41000, "water": "High", "sustain": "Medium"},
                "Maize": {"yield": 2200, "profit": 27000, "water": "Low", "sustain": "High"},
            }

            st.subheader("✅ Recommended Crops for You")
            for idx, crop in enumerate(recommended):
                details = info_map.get(crop, {"yield": "N/A", "profit": "N/A", "water": "N/A", "sustain": "N/A"})
                st.markdown(f"""
                    <div style="
                        padding:16px;
                        margin-bottom:14px;
                        border-radius:12px;
                        background:linear-gradient(135deg,#ffffff,#f1fff1);
                        box-shadow:0 6px 20px rgba(0,0,0,0.06);">
                        <h3 style="margin:4px 0;">🌱 {crop}</h3>
                        <p style="margin:4px 0;">🌾 <b>Expected Yield:</b> {details['yield']} kg/acre</p>
                        <p style="margin:4px 0;">💰 <b>Profit Potential:</b> ₹{details['profit']}</p>
                        <p style="margin:4px 0;">💧 <b>Water Requirement:</b> {details['water']}</p>
                        <p style="margin:4px 0;">🌍 <b>Sustainability:</b> {details['sustain']}</p>
                    </div>
                """, unsafe_allow_html=True)

    # ---------------- Ask Assistant ----------------
    elif feature == "Ask Assistant":
        st.subheader("🤖 Ask Kisan Assistant — Chat with AI")
        st.write("Ask farming-related queries and receive an instant, conversational answer.")

        import os
        try:
            import openai
        except ImportError:
            st.error("The `openai` package is not installed. Run `pip install openai` and restart the app.")
            st.stop()

        if "openai_api_key" not in st.session_state:
            st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")

        if not st.session_state.openai_api_key:
            api_key_input = st.text_input(
                "Enter OpenAI API Key (session only)", type="password", key="assistant_api_key"
            )
            if api_key_input:
                st.session_state.openai_api_key = api_key_input

        if not st.session_state.openai_api_key:
            st.warning("You need to provide an OpenAI API key to use the assistant.")
            st.stop()

        openai.api_key = st.session_state.openai_api_key

        if "assistant_messages" not in st.session_state:
            st.session_state.assistant_messages = [
                {"role": "system", "content": (
                    "You are KiSAN Sahayak assistant — a helpful, concise, farmer-focused assistant. "
                    "Give actionable, practical advice, short steps, include safety/timing notes, "
                    "ask clarifying questions only when strictly necessary."
                )}
            ]
            st.session_state.chat_history = []

        user_input = st.text_input("💬 Type your question here...", key="assistant_user_input")
        if st.button("Ask", key="assistant_ask_btn") and user_input.strip():
            st.session_state.assistant_messages.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append(("user", user_input))

            # Call OpenAI API
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=st.session_state.assistant_messages,
                    max_tokens=500,
                    temperature=0.2
                )
                assistant_reply = response["choices"][0]["message"]["content"].strip()
            except Exception as e:
                assistant_reply = f"⚠️ API Error: {e}"

            st.session_state.assistant_messages.append({"role": "assistant", "content": assistant_reply})
            st.session_state.chat_history.append(("assistant", assistant_reply))

        # Render chat
        for role, msg in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"""
                    <div style="display:flex; justify-content:flex-end; margin:6px 0;">
                        <div style="background:#e6f4ea; padding:12px; border-radius:12px; max-width:75%; text-align:right;">
                            <b>👩‍🌾 You</b><br>{msg}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="display:flex; justify-content:flex-start; margin:6px 0;">
                        <div style="background:#f0f7ff; padding:12px; border-radius:12px; max-width:75%; text-align:left;">
                            <b>🤖 Assistant</b><br>{msg}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        st.caption("Tip: ask practical questions like 'Which crop suits 5 acres in Pune in September?' or 'How to treat powdery mildew on tomatoes?'.")

    # ---------------- Alerts & Highlights ----------------
elif features == "Alerts & Highlights":
    st.subheader("⚡ Alerts & Highlights")
    st.write("Important updates about weather, pests, market and crop health.")
    st.info("No active alerts (demo).")
    st.write("- Market: Rice prices +3% (demo)")
    st.write("- Weather: Light showers expected tomorrow (demo)")           
# ---------------- Plant Helpdesk ----------------
elif features == "Plant Helpdesk":
    st.subheader("📚 Plant Helpdesk")
    st.write("Learn about growing a crop from seed to harvest.")
    plant_name = st.text_input("Enter plant name (e.g., Mango, Tomato)", key="helpdesk_input")
    if st.button("Show Guide", key="helpdesk_show"):
        if plant_name.strip():
            st.success(f"Guide for {plant_name} (demo).")
            st.markdown(f"**{plant_name}** — basic steps (demo):\n\n1. Select variety\n2. Prepare soil\n3. Sow/Transplant\n4. Irrigate & fertilize\n5. Monitor pests\n6. Harvest")
        else:
            st.warning("Please enter a plant name.")