from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from sentence_transformers import SentenceTransformer, util
import time
import random
import json

app = Flask(__name__)
socketio = SocketIO(app)

# Load the language model and embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load emergency data from JSON file with error handling
try:
    with open('emergency_data.json', 'r') as f:
        emergency_data = json.load(f).get('emergencies', [])
except FileNotFoundError:
    print("Error: 'emergency_data.json' file not found.")
    emergency_data = []
except json.JSONDecodeError:
    print("Error: Failed to decode JSON from 'emergency_data.json'.")
    emergency_data = []

# Generate embeddings for emergency keywords
emergency_embeddings = []
for emergency in emergency_data:
    keywords_text = " ".join(emergency.get('keywords', []))
    emergency_embeddings.append(embedding_model.encode(keywords_text))

# Track conversation state
conversation_state = {}

def find_best_match(user_input):
    user_embedding = embedding_model.encode(user_input)
    similarities = [util.cos_sim(user_embedding, embedding)[0][0].item() for embedding in emergency_embeddings]
    best_match_idx = similarities.index(max(similarities))
    return emergency_data[best_match_idx], max(similarities)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('user_message')
def handle_message(data):
    user_input = data['message'].lower()
    session_id = data.get('session_id')  # Assuming you track session_id for different users
    
    if session_id not in conversation_state:
        conversation_state[session_id] = {'step': 1}

    state = conversation_state[session_id]

    if state['step'] == 1:
        emit('bot_response', {"message": "Are you having an emergency, or would you like to leave a message?"})
        state['step'] = 2
    
    elif state['step'] == 2:
        if "emergency" in user_input:
            emit('bot_response', {"message": "Please describe the emergency."})
            state['step'] = 3
        elif "message" in user_input:
            emit('bot_response', {"message": "Please leave your message."})
            state['step'] = 6
        else:
            emit('bot_response', {"message": "I don’t understand that. Are you having an emergency, or would you like to leave a message?"})

    elif state['step'] == 3:
        if emergency_data:
            best_match, similarity_score = find_best_match(user_input)

            # Setting a lower threshold and more explicit matching mechanism
            if similarity_score < 0.5:
                emit('bot_response', {"message": "I'm not sure I understand. Could you please describe the situation in more detail?"})
            else:
                emit('bot_response', {"message": "I am checking what you should do immediately. Meanwhile, can you tell me which area you are located right now?"})
                state['best_match'] = best_match
                state['step'] = 4
        else:
            emit('bot_response', {"message": "Emergency data is not available. Please try again later."})

    elif state['step'] == 4:
        location = user_input
        state['location'] = location
        eta = random.randint(5, 15)
        emit('bot_response', {"message": f"Dr. Adrin will arrive at {location} in approximately {eta} minutes."})
        state['step'] = 5

    elif state['step'] == 5:
        time.sleep(15)  # Simulating delay
        best_match = state.get('best_match')
        emit('bot_response', {"message": f"I found a match: {best_match['name']}. {best_match['steps']} Dr. Adrin is on the way."})
        emit('bot_response', {"message": "Don't worry, please follow these steps, Dr. Adrin will be with you shortly."})
        del conversation_state[session_id]  # Reset conversation state for the session

    elif state['step'] == 6:
        emit('bot_response', {"message": "Thanks for the message, we will forward it to Dr. Adrin."})
        del conversation_state[session_id]  # Reset conversation state for the session

if __name__ == "__main__":
    socketio.run(app, debug=True)
