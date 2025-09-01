import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import re
import unicodedata
from typing import Dict, Any, List

# =========================
# App Config
# =========================
st.set_page_config(
    page_title="Food Tourism Assistant Indonesia 🇮🇩",
    page_icon="🍜",
    layout="centered"
)
st.title("🍜 Food Tourism Assistant Indonesia")
st.caption("Explore Indonesian cuisine with AI-powered recognition and cultural insights.")

# tombol popover di area utama
with st.popover("📖 How to use the chatbot"):
    st.markdown("""
    ### 📖 How to use
    1. 📷 **Upload a food photo first** so the chatbot can recognize the dish.  
    2. 🤖 The chatbot will identify the dish and give some information.  
    3. 💬 After the dish is detected, you can continue the conversation by asking:  
       - Where can I find this food?  
       - How spicy is it?  
       - Is it halal or not?  
       - What’s its cultural or historical background?  
       - What are the nutrition facts?  
    4. 📝 You can ask in **English or Bahasa Indonesia**.  
    """)
    

# =========================
# Load Assets
# =========================
@st.cache_resource
def load_interpreter():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource
def load_labels() -> List[str]:
    with open("labels.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

@st.cache_resource
def load_food_data() -> Dict[str, Any]:
    with open("foods.json", "r", encoding="utf-8") as f:
        return json.load(f)

interpreter = load_interpreter()
labels = load_labels()
food_db = load_food_data()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def _input_size():
    # infer HxW from model input shape, e.g. (1,224,224,3) or (1,3,224,224)
    shape = input_details[0]["shape"]
    if len(shape) == 4:
        if shape[1] in (1,3) and shape[2] > 3:  # NCHW
            return int(shape[2]), int(shape[3]), "NCHW"
        else:  # NHWC
            return int(shape[1]), int(shape[2]), "NHWC"
    return 224, 224, "NHWC"

IMG_H, IMG_W, LAYOUT = _input_size()

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize((IMG_W, IMG_H))
    arr = np.asarray(img, dtype=np.float32)
    # Normalize to [0,1] if model expects float
    if input_details[0]["dtype"] == np.float32:
        arr = arr / 255.0
    # Layout
    if LAYOUT == "NHWC":
        arr = np.expand_dims(arr, axis=0)
    else:  # NCHW
        arr = np.transpose(arr, (2,0,1))
        arr = np.expand_dims(arr, axis=0)
    return arr.astype(input_details[0]["dtype"])

def predict(image: Image.Image):
    x = preprocess(image)
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx]) if preds.ndim == 1 else float(np.max(preds))
    return labels[idx], conf

# =========================
# Utils (Text & Safety)
# =========================
def normalize(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    return s

def exists_field(info: Dict[str, Any], key: str) -> bool:
    return key in info and str(info[key]).strip() != ""

def safe_get(info: Dict[str, Any], key: str, default: str = "-") -> str:
    return str(info.get(key, default)).strip() or default

# =========================
# Rule-based NLU (Intent Detection)
# =========================
def detect_intents(q: str) -> List[str]:
    qn = normalize(q)
    intents = []

    # comparison to western/italian meatballs
    if any(k in qn for k in ["compare", "different", "difference", "italian", "western", "why different", "beda", "banding"]):
        intents.append("compare_western")

    # street vs home eating context
    if any(k in qn for k in ["street", "cart", "kaki lima", "warung", "home", "rumah", "outside", "di luar", "di jalan"]):
        intents.append("street_vs_home")

    # price / cheap / expensive / cost jakarta
    if any(k in qn for k in ["price", "how much", "cost", "expensive", "cheap", "harga", "berapa", "jakarta"]):
        intents.append("price")

    # cultural meaning / event
    if any(k in qn for k in ["culture", "cultural", "event", "special", "tradition", "budaya", "acara"]):
        intents.append("culture_events")

    # spicy level
    if any(k in qn for k in ["spicy", "pedas", "chili", "sambal", "how spicy"]):
        intents.append("spice")

    # halal
    if "halal" in qn:
        intents.append("halal")

    # meat type / ingredients
    if any(k in qn for k in ["meat", "beef", "chicken", "pork", "daging", "bahan", "what meat"]):
        intents.append("meat")

    # nutrition: calories/protein
    if any(k in qn for k in ["calorie", "calories", "kcal", "protein", "nutrition", "nutrisi"]):
        intents.append("nutrition")

    # vegetarian / vegan / chicken version / variants
    if any(k in qn for k in ["vegetarian", "vegan", "chicken version", "chicken only", "varian", "variant", "versions"]):
        intents.append("variants")

    # gluten
    if any(k in qn for k in ["gluten", "gluten-free", "bebas gluten"]):
        intents.append("gluten")

    # snack vs main
    if any(k in qn for k in ["snack", "main", "meal", "cemilan", "makanan utama"]):
        intents.append("meal_type")

    # how to order bahasa
    if any(k in qn for k in ["how to order", "order", "pesan", "bahasa", "say", "cara pesan"]):
        intents.append("ordering")

    # safety for foreigners’ stomach / hygiene
    if any(k in qn for k in ["safe", "safety", "stomach", "hygiene", "aman", "higienis", "risk", "risky"]):
        intents.append("safety")

    # meal time (morning lunch night)
    if any(k in qn for k in ["morning", "breakfast", "lunch", "dinner", "malam", "siang", "pagi"]):
        intents.append("meal_time")

    # where to try (generic)
    if any(k in qn for k in ["where", "where to try", "where can i", "di mana", "dimana"]):
        intents.append("where")

    return list(dict.fromkeys(intents))  # unique preserve order

# =========================
# Response Generator (Rule-based)
# =========================
def answer_by_intent(food: str, q: str, info: Dict[str, Any]) -> str:
    intents = detect_intents(q)
    ans_parts = []

    # fallback to general description if no specific intent matched
    if not intents:
        # small heuristic: if user asks "what is this"
        if re.search(r"\bwhat is (this|it)\b", normalize(q)) or "what name" in normalize(q):
            return f"This is **{food}**. {safe_get(info, 'one_liner', safe_get(info, 'description', 'A popular Indonesian food.'))}"
        return f"About **{food}**: {safe_get(info, 'description', safe_get(info, 'one_liner', '-'))}"

    for intent in intents:
        if intent == "compare_western":
            if exists_field(info, "comparison"):
                ans_parts.append(safe_get(info, "comparison"))
            else:
                ans_parts.append(f"Compared to Western versions, **{food}** is usually served with Indonesian-style seasonings and sides, often in soup or with sambal.")

        elif intent == "street_vs_home":
            ans_parts.append(safe_get(info, "street_vs_home", "Common both at home and from street vendors (kaki lima)."))

        elif intent == "price":
            ans_parts.append(safe_get(info, "price_range", "Prices vary by city and stall; street versions are usually affordable."))

        elif intent == "culture_events":
            ans_parts.append(safe_get(info, "cultural_meaning", "It’s an everyday comfort food rather than a ceremonial dish."))

        elif intent == "spice":
            ans_parts.append(safe_get(info, "spice_level", "Spice level ranges from mild to hot; ask for less sambal if sensitive."))

        elif intent == "halal":
            ans_parts.append(safe_get(info, "halal_info", "Many places offer halal options; ask the vendor to be sure."))

        elif intent == "meat":
            base = safe_get(info, "meat_info", safe_get(info, "ingredients", "-"))
            ans_parts.append(base)

        elif intent == "nutrition":
            ans_parts.append(safe_get(info, "nutrition", "Calories and protein vary by portion and vendor."))

        elif intent == "variants":
            ans_parts.append(safe_get(info, "variants", "There are regional and modern variations."))

        elif intent == "gluten":
            ans_parts.append(safe_get(info, "gluten_info", "If you avoid gluten, ask for modifications and avoid wheat-based sides."))

        elif intent == "meal_type":
            ans_parts.append(safe_get(info, "meal_type", "Enjoyed as a main meal or hearty snack."))

        elif intent == "ordering":
            ans_parts.append(safe_get(info, "ordering_indo", "") + "\n" + safe_get(info, "ordering_tips", ""))

        elif intent == "safety":
            ans_parts.append(safe_get(info, "safety", "Choose busy, clean stalls; hot, freshly cooked servings are safer."))

        elif intent == "meal_time":
            ans_parts.append(safe_get(info, "meal_time", "Common for lunch and dinner, but available all day."))

        elif intent == "where":
            ans_parts.append(safe_get(info, "where_to_try", "Look for popular warungs or busy street carts (kaki lima)."))

    return "\n\n".join([p for p in ans_parts if p.strip()])

# =========================
# Session State
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.current_food = None
    st.session_state.conf = 0.0

    # Greeting awal
    reply1 = (
        "👋 Hi! I'm your Indonesian Food Tourism Assistant.\n\n"
        "Please upload a photo of the dish first so I can recognize it. "
        "After that, you can ask more specific questions."
    )
    reply2 = (
        "💡 Once a dish is identified, you can ask things like:\n"
        "- Where can I find this food?\n"
        "- How spicy is it?\n"
        "- Is it halal?\n"
        "- What’s its cultural background?\n"
        "- What are the nutrition facts?\n"
    )
    st.session_state.messages.append({"role": "assistant", "content": reply1})
    st.session_state.messages.append({"role": "assistant", "content": reply2})

# =========================
# Chat History
# =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "image":
            st.image(msg["content"], caption="Uploaded Image", use_column_width=True)
        else:
            st.write(msg["content"])

# =========================
# Chat Input
# =========================
if st.session_state.current_food:
    user_q = st.chat_input("Ask me anything about the dish...")

    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})

        food = st.session_state.current_food
        info = food_db.get(food, {})
        reply = answer_by_intent(food, user_q, info)
        st.session_state.messages.append({"role": "assistant", "content": reply})

        st.rerun()
else:
    st.info("📷 Please upload a food photo first to start chatting.")

# =========================
# UI — Upload & Classify
# =========================
uploaded = st.file_uploader("Upload a food photo", type=["jpg","jpeg","png"])

if uploaded and st.session_state.current_food is None:
    image = Image.open(uploaded).convert("RGB")

    # Simpan ke chat history sebagai bubble image
    st.session_state.messages.append({
        "role": "user",
        "type": "image",
        "content": image
    })

    # klasifikasi
    label, conf = predict(image)

    if conf < 0.3:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"🤔 I'm not confident enough to identify this dish (confidence {conf:.0%})."
        })
    else:
        st.session_state.current_food = label
        st.session_state.conf = conf
        info = food_db.get(label, {})

        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                f"✅ Identified as **{label}** (Confidence {conf:.0%}).\n\n"
                f"{safe_get(info, 'one_liner', safe_get(info, 'description', ''))}\n\n"
                f"**Origin:** {safe_get(info, 'origin')}\n"
                f"\n\n**Ingredients:** {safe_get(info, 'ingredients')}\n"
                f"\n\n**Taste:** {safe_get(info, 'taste')}\n\n"
                f"Now you can ask me about {label} like: comparison, culture, price, safety, "
                f"how to order, nutrition, variants, gluten and halal."
            )
        })

    st.rerun()
