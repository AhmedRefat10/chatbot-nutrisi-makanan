The Food Tourism Assistant app, based on Streamlite, helps foreign tourists learn about Indonesian cuisine.

Users can upload food photos, and the app will recognize the food using a TensorFlow Lite model. The chatbot will then provide cultural explanations, key ingredients, and tips for enjoying the food based on a JSON database.

## ✨ Key Features

- 📷 **Upload Food Photos** → Predicts typical Indonesian dishes using the TFLite model.
- 🧑‍🍳 **Culinary Explanations** → Displays descriptions, key ingredients, flavors, and cultural context from JSON.
- 💬 **Interactive Chatbot** → Tourists can ask questions about food, preparation methods, and culinary recommendations.
- 💡 **Chat Bubble UI** → Conversation interface similar to a chat app.
- 🕒 **Chat History** → Saves conversation history during a session.

## 🛠️ Technologies Used

- **Python 3.x**
- **Streamlit**
- **TensorFlow Lite**
- **NumPy** for data processing
- **JSON** (Indonesian food database)

## 📂 Project Structure

```
├── app.py # Main application script
├── foods.json # Nutritional data per 100 grams
├── model.tflite # TensorFlow Lite model
├── labels.txt # Food class labels
├── requirements.txt # Project dependencies
├── README.md # Project documentation
```

## 🚀 How to Run

1. **Clone this repository**
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Streamlit app:
```bash
streamlit run app.py
```

## 📌 How to Use

1. Upload a food image via the upload panel.
2. See automatic food predictions and nutritional information.
3. Use the chatbot to ask for more detailed information about the food's nutritional content.
4. The conversation history will appear on the screen.

## 📝 Notes

- Indonesian culinary data is stored in `foods.json`.
- The TFLite model must match the labels in `labels.txt`.
- The application is intended to help foreign tourists understand Indonesian cuisine.

## 📜 License

This project was created for the purposes of learning, research, and promoting Indonesian culinary tourism.
