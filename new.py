from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer
import torch
import time


app = Flask(__name__)

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models and move them to the GPU
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
intent_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
intent_model.to(device)  # Move model to GPU

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # SentenceTransformer automatically handles device placement

# Dictionary to store emergency instructions
emergencies = {
    "not breathing": "Please start CPR immediately. CPR involves pushing against the chest of the patient and blowing air into their mouth in a constant rhythm.",
    "chest pain": "Please have the patient sit down and try to relax. Avoid any physical activity until Dr. Adrin arrives.",
    "unconscious": "Check if the patient is breathing and try to wake them up by gently shaking them or calling their name loudly."
}

# Function to classify intent using BERT model
def classify_intent(text):
    # Tokenize the input text and move it to the GPU
    inputs = tokenizer(text, return_tensors="pt", clean_up_tokenization_spaces=True).to(device)
    with torch.no_grad():
        outputs = intent_model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()

    # Convert predictions to labels (Assuming LABEL_0 corresponds to emergency, LABEL_1 to message)
    if predictions[0] == 0:
        return "LABEL_0"
    elif predictions[0] == 1:
        return "LABEL_1"
    else:
        return "UNKNOWN"

def handle_emergency(emergency_details):
    # Simple lookup in the dictionary
    for emergency, instructions in emergencies.items():
        if emergency in emergency_details.lower():
            return instructions
    return "I'm sorry, I don't have instructions for that specific emergency."

def handle_conversation(user_input):
    intent = classify_intent(user_input)

    if intent == "LABEL_0":  # Assuming LABEL_0 corresponds to "emergency"
        emergency_details = user_input
        # Respond while processing the emergency
        time.sleep(15)  # Artificial delay to simulate processing
        instructions = handle_emergency(emergency_details)
        return f"I am checking what you should do immediately, meanwhile, can you tell me which area you are located in? Instructions: {instructions}"
    
    elif intent == "LABEL_1":  # Assuming LABEL_1 corresponds to "message"
        return "Please provide your message."
    
    else:
        return "I don't understand that. Are you having an emergency or would you like to leave a message?"

@app.route('/ai-receptionist', methods=['POST'])
def ai_receptionist():
    user_input = request.json.get('message')
    response = handle_conversation(user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
