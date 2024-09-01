from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import json
import time
import uuid
from qdrant_client.models import PointStruct, VectorParams, Distance

app = Flask(__name__)

# Load models
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
intent_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Qdrant client
client = QdrantClient(host="localhost", port=6333)  # Adjust as necessary

nlp_pipeline = pipeline("text-classification", model=intent_model, tokenizer=tokenizer)

# Function to classify intent using BERT model
def classify_intent(text):
    result = nlp_pipeline(text)[0]
    intent = result['label']
    return intent

# Function to vectorize text using Sentence-BERT
def vectorize_text(text):
    embedding = embedding_model.encode(text, convert_to_tensor=True).cpu().numpy()
    return embedding

def load_and_process_data(json_file):
    vector_size = 384  # The vector size should match the output size of your SentenceTransformer model
    distance = Distance.COSINE  # Using cosine distance for similarity

    # Check if the collection already exists
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    if "emergencies" not in collection_names:
        # Create the collection if it does not exist
        client.create_collection(
            collection_name="emergencies",
            vectors_config=VectorParams(size=vector_size, distance=distance)
        )
        print("Collection `emergencies` created.")
    else:
        print("Collection `emergencies` already exists.")

    with open(json_file, 'r') as f:
        emergency_data = json.load(f)

    for entry in emergency_data:
        emergency_vector = vectorize_text(entry['emergency'])  # Convert to vector using Sentence-BERT
        point_id = str(uuid.uuid4())  # Generate a unique UUID for each point
        client.upsert(
            collection_name="emergencies",
            points=[
                PointStruct(
                    id=point_id,  # Use the generated UUID as the unique identifier
                    vector=emergency_vector.tolist(),
                    payload={
                        "emergency": entry['emergency'],  # Store the emergency text as part of the payload instead
                        "instructions": entry['instructions'],
                        "source": entry['source']
                    }
                )
            ]
        )


def handle_emergency(emergency_details):
    emergency_vector = vectorize_text(emergency_details)
    search_result = client.search(
        collection_name="emergencies",  # Corrected to match the collection name
        query_vector=emergency_vector.tolist(),
        limit=1
    )

    if search_result:
        return search_result[0].payload['instructions']  # Ensure correct access to payload data
    else:
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
    load_and_process_data('emergency_data.json')
    app.run(debug=True)


