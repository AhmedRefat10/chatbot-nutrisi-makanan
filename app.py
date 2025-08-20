import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import random

# Load labels
with open("labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]

# Load food data JSON
with open("food_data.json", "r", encoding="utf-8") as f:
    food_data = json.load(f)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(image):
    img = image.resize((224, 224))  # sesuaikan ukuran model
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = np.argmax(preds)
    return labels[idx], preds[idx]

# Fun facts database
fun_facts = [
    "Did you know? Satay was once listed among the world's most delicious foods by CNN Travel!",
    "Fun fact: In Padang restaurants, all dishes are displayed on the table, but you only pay for what you eat!",
    "Many Indonesian foods are traditionally eaten with hands 鈥� it鈥檚 part of the culture!",
    "Spices in Indonesian cuisine reflect the country鈥檚 history as a major spice trading hub."
]

# Streamlit App
st.set_page_config(page_title="Food Tourism Assistant 馃嚠馃嚛", page_icon="馃崪")
st.title("馃崪 Food Tourism Assistant Indonesia 馃嚠馃嚛")
st.write("Upload a food photo and chat with your culinary guide!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_food" not in st.session_state:
    st.session_state.current_food = None

# Upload food photo
uploaded = st.file_uploader("Upload a food photo", type=["jpg", "jpeg", "png"])

if uploaded and st.session_state.current_food is None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    label, conf = predict(image)
    st.session_state.current_food = label

    if label in food_data:
        info = food_data[label]
        intro = (
            f"This is **{label}**, a dish from {info['origin']}. 馃嵔锔廫n\n"
            f"It鈥檚 made with {info['ingredients']}, usually tastes **{info['taste']}**, "
            f"and is often enjoyed like this: {info['culture']}.\n\n"
            f"Tips for you: {info['tips']}.\n\n"
            f"{info['description']}\n\n"
            f"鉁� {random.choice(fun_facts)}"
        )
        st.session_state.messages.append({"role": "assistant", "content": intro})

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# Chat input
if st.session_state.current_food:
    prompt = st.chat_input("Ask about this food!")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get food info
        food = st.session_state.current_food
        info = food_data.get(food, {})

        # Simple rule-based responses
        response = ""
        q = prompt.lower()
        if "spicy" in q:
            if "馃尪锔�" in info.get("tips", "") or "spicy" in info.get("taste", "").lower():
                response = f"Yes, **{food}** can be quite spicy! 馃尪锔� You might want to start with a mild version."
            else:
                response = f"**{food}** is generally not very spicy, but it depends on how it鈥檚 cooked."
        elif "where" in q:
            response = f"You can try delicious **{food}** in many local warungs (small restaurants). "
            response += "If you're in Jakarta, I recommend going to traditional food stalls for the most authentic taste!"
        elif "how" in q and "eat" in q:
            response = f"Locals usually eat **{food}** like this: {info.get('culture', 'simply served with rice or as a snack')}."
        else:
            response = f"Here鈥檚 more about **{food}**: {info.get('description', 'A tasty Indonesian dish!')}"

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
