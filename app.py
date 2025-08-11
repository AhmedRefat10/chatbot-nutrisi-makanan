import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import re

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

# ---------- Load Labels & Nutrition ----------
@st.cache_data
def load_labels():
    with open("labels.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

@st.cache_data
def load_nutrition():
    with open("nutrition.json", "r") as f:
        return json.load(f)

# ---------- Prediction ----------
def predict_image(image, interpreter, labels):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    if input_details[0]['dtype'] == np.float32:
        img_array = img_array / 255.0

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    top_index = np.argmax(output_data)
    return labels[top_index], float(output_data[top_index])

# ---------- Get Nutrition ----------
def get_nutrition(item, berat, use_portion):
    if item["type"] == "buah-sayur":
        cal = item["nutrition_per_100g"]["calories"] * berat / 100
        prot = item["nutrition_per_100g"]["proteins"] * berat / 100
        fat = item["nutrition_per_100g"]["fat"] * berat / 100
        carb = item["nutrition_per_100g"]["carbohydrate"] * berat / 100
        source = f"per 100 gram (buah/sayur)"
    else:
        if use_portion:
            cal = item["nutrition_per_portion"]["calories"]
            prot = item["nutrition_per_portion"]["proteins"]
            fat = item["nutrition_per_portion"]["fat"]
            carb = item["nutrition_per_portion"]["carbohydrate"]
            source = f"per porsi ({item['portion_size_g']} gram)"
        else:
            cal = item["nutrition_per_100g"]["calories"] * berat / 100
            prot = item["nutrition_per_100g"]["proteins"] * berat / 100
            fat = item["nutrition_per_100g"]["fat"] * berat / 100
            carb = item["nutrition_per_100g"]["carbohydrate"] * berat / 100
            source = f"per 100 gram"
    return cal, prot, fat, carb, source

# ---------- Chatbot Response ----------
def chatbot_response(user_input, last_prediction, nutrition_data, berat, use_portion):
    if not last_prediction:
        return ["Silakan kirim gambar makanan terlebih dahulu."]

    berat_chat = berat
    porsi_count = 1

    porsi_match = re.search(r"(\d+)\s*porsi", user_input.lower())
    if porsi_match:
        porsi_count = int(porsi_match.group(1))

    berat_match = re.search(r"(\d+)\s*(gram|gr|g)", user_input.lower())
    if berat_match:
        berat_chat = int(berat_match.group(1))

    for item in nutrition_data:
        if item["name"].lower() == last_prediction.lower():
            cal, prot, fat, carb, source = get_nutrition(item, berat_chat, use_portion)

            if use_portion and porsi_count > 1 and item["type"] == "non-buah-sayur":
                cal *= porsi_count
                prot *= porsi_count
                fat *= porsi_count
                carb *= porsi_count
                source = f"per {porsi_count} porsi ({item['portion_size_g']} gram x {porsi_count})"
            elif not use_portion:
                if porsi_count > 1:
                    berat_chat *= porsi_count
                    cal, prot, fat, carb, source = get_nutrition(item, berat_chat, use_portion)

            responses = []
            responses.append(f"Sepertinya ini adalah {last_prediction}.")

            lower_input = user_input.lower()
            if "kalori" in lower_input:
                responses.append(f"Kandungan kalori pada makanan tersebut adalah sekitar {cal:.2f} kcal untuk {source}.")
            elif "protein" in lower_input:
                responses.append(f"Kandungan protein pada makanan tersebut adalah sekitar {prot:.2f} gram untuk {source}.")
            elif "lemak" in lower_input:
                responses.append(f"Kandungan lemak pada makanan tersebut adalah sekitar {fat:.2f} gram untuk {source}.")
            elif "karbo" in lower_input or "karbohidrat" in lower_input:
                responses.append(f"Kandungan karbohidrat pada makanan tersebut adalah sekitar {carb:.2f} gram untuk {source}.")
            else:
                responses.append(
                    f"Data gizi {last_prediction} untuk {source} adalah:\n"
                    f"- Kalori: {cal:.2f} kcal\n"
                    f"- Protein: {prot:.2f} g\n"
                    f"- Lemak: {fat:.2f} g\n"
                    f"- Karbohidrat: {carb:.2f} g"
                )
            return responses
    return ["Maaf, data nutrisi tidak ditemukan."]

# ---------- Render Chat Bubble ----------
def render_message(msg):
    sender = msg["sender"]
    content = msg["content"]
    is_image = msg.get("is_image", False)

    if sender == "user":
        if is_image:
            st.markdown('<div style="text-align: right; font-weight:bold; color:#064273;">Anda (Foto):</div>', unsafe_allow_html=True)
            st.image(content, width=250)
        else:
            st.markdown(
                f'<div style="text-align: right; background-color: #1E90FF; color: white; padding: 10px; '
                f'border-radius: 15px; margin: 6px 0; max-width: 70%; margin-left: auto; font-size: 16px;">{content}</div>', 
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            f'<div style="text-align: left; background-color: #f1f0f0; color: #333; padding: 10px; '
            f'border-radius: 15px; margin: 6px 0; max-width: 70%; font-size: 16px;">{content}</div>', 
            unsafe_allow_html=True
        )

# ---------- Main App ----------
st.set_page_config(page_title="Chatbot Estimasi Kalori", layout="wide")
st.title("🍽 Chatbot Estimasi Kalori Makanan (Chat Mode)")

interpreter = load_model()
labels = load_labels()
nutrition_data = load_nutrition()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_weight" not in st.session_state:
    st.session_state.last_weight = 100
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0

weight = st.sidebar.number_input("Masukkan berat makanan (gram)", min_value=1, value=st.session_state.last_weight, step=1)
st.session_state.last_weight = weight

per_100g = st.sidebar.checkbox("Tampilkan nilai gizi per 100 gram", value=True)
per_portion = st.sidebar.checkbox("Tampilkan nilai gizi per porsi (untuk makanan non buah & sayur)", value=False)

st.subheader("Chat")

for msg in st.session_state.messages:
    render_message(msg)

st.markdown("---")

col1, col2 = st.columns([3,1])
with col1:
    user_text = st.text_input("Ketik pesan di sini...", key="input_text")
    submit = st.button("Kirim")
with col2:
    user_img = st.file_uploader(
        "Upload gambar makanan",
        type=["jpg","jpeg","png"],
        key=f"uploaded_file_{st.session_state.upload_key}"
    )

def add_message(content, sender="user", is_image=False):
    st.session_state.messages.append({"content": content, "sender": sender, "is_image": is_image})

# Handle upload gambar
if user_img is not None:
    img = Image.open(user_img).convert("RGB")
    add_message(img, sender="user", is_image=True)

    pred_label, confidence = predict_image(img, interpreter, labels)
    st.session_state.last_prediction = pred_label

    matched_item = next((x for x in nutrition_data if x["name"].lower() == pred_label.lower()), None)

    if matched_item:
        cal, prot, fat, carb, source = get_nutrition(matched_item, st.session_state.last_weight, per_portion)
        nutri_text = (
            f"Hasil analisis gambar menunjukkan makanan adalah {pred_label}.\n\n"
            f"Kandungan gizinya sekitar:\n"
            f"- Kalori: {cal:.2f} kcal ({source})\n"
            f"- Protein: {prot:.2f} g\n"
            f"- Lemak: {fat:.2f} g\n"
            f"- Karbohidrat: {carb:.2f} g"
        )
    else:
        nutri_text = "Data nutrisi tidak ditemukan."

    add_message(nutri_text, sender="bot")
    st.session_state.upload_key += 1

# Handle user text submit
if submit and user_text:
    add_message(user_text, sender="user")

    bot_responses = chatbot_response(
        user_text,
        st.session_state.last_prediction,
        nutrition_data,
        st.session_state.last_weight,
        per_portion and not per_100g
    )
    for resp in bot_responses:
        add_message(resp, sender="bot")

    # Reset input text after submit
    st.session_state["input_text"] = ""

