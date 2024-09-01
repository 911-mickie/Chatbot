from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import json
import torch
from qdrant_client.models import VectorParams, Distance

# Load models
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
intent_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Qdrant client

client = QdrantClient(host="localhost", port=6333)  # Adjust as necessary

# qdrant_client = QdrantClient(url="http://localhost:6333")
# if not client.collection_exists("emergencies"):
#     client.create_collection(
#         collection_name="emergencies",
#         vectors_config=VectorParams(size=384, distance=Distance.COSINE),  # Assuming your vector size is 384 (Sentence-BERT output)
#     )

# Intent classification pipeline
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

# Load and process the curated data
# def load_and_process_data(json_file):
#     with open(json_file, 'r') as f:
#         emergency_data = json.load(f)

#     for entry in emergency_data:
#         emergency_vector = vectorize_text(entry['emergency'])

#         qdrant_client.upsert(
#             collection_name="emergencies",
#             points=[
#                 models.PointStruct(
#                     id=entry['emergency'],
#                     vector=emergency_vector.tolist(),
#                     payload={
#                         "instructions": entry['instructions'],
#                         "source": entry['source']
#                     }
#                 )
#             ]
#         )

# # Assuming the curated data is saved in 'emergency_data.json'
# load_and_process_data('emergency_data.json')

from qdrant_client.models import PointStruct
import numpy as np

def load_and_process_data(json_file):
    with open(json_file, 'r') as f:
        emergency_data = json.load(f)

    for entry in emergency_data:
        emergency_vector = vectorize_text(entry['emergency'])  # Convert to vector using Sentence-BERT
        client.upsert(
            collection_name="emergencies",
            points=[
                PointStruct(
                    id=entry['emergency'],  # Use a unique identifier or a hash of the emergency description
                    vector=emergency_vector.tolist(),
                    payload={
                        "instructions": entry['instructions'],
                        "source": entry['source']
                    }
                )
            ]
        )


# Function to handle emergency by querying the vector database
# def handle_emergency(emergency_details):
#     emergency_vector = vectorize_text(emergency_details)
#     search_result = qdrant_client.search(
#         collection_name="emergencies",
#         query_vector=emergency_vector.tolist(),
#         limit=1
#     )
    
#     if search_result:
#         return search_result[0].payload['instructions']
#     else:
#         return "I'm sorry, I don't have instructions for that specific emergency."


def handle_emergency(emergency_details):
    emergency_vector = vectorize_text(emergency_details)
    search_result = client.search(
        collection_name="emergencies",
        query_vector=emergency_vector.tolist(),
        limit=1
    )

    if search_result:
        return search_result[0]['payload']['instructions']  # Assuming 'payload' contains 'instructions'
    else:
        return "I'm sorry, I don't have instructions for that specific emergency."



# Example conversation flow using the enhanced system
def handle_conversation(user_input):
    intent = classify_intent(user_input)

    if intent == "LABEL_0":  # Assuming LABEL_0 corresponds to "emergency"
        emergency_details = user_input  # In a real scenario, further entity extraction would refine this
        print("I am checking what you should do immediately, meanwhile, can you tell me which area are you located right now?")
        instructions = handle_emergency(emergency_details)
        print("Instructions:", instructions)
        return instructions
    
    elif intent == "LABEL_1":  # Assuming LABEL_1 corresponds to "message"
        return "Please provide your message."
    
    else:
        return "I don't understand that. Are you having an emergency or would you like to leave a message?"

# Example usage
user_input = "My friend is having heart attack symptoms."
response = handle_conversation(user_input)
print(response)
