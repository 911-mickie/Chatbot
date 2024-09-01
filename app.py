from flask import Flask, request, jsonify, render_template
import time
import random

app = Flask(__name__)

emergencies = {
    "not breathing": "Please start CPR immediately. CPR involves pushing against the chest of the patient and blowing air into their mouth in a constant rhythm.",
    "chest pain": "Please have the patient sit down and try to relax. Avoid any physical activity until Dr. Adrin arrives.",
    "unconscious": "Check if the patient is breathing and try to wake them up by gently shaking them or calling their name loudly."
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ai-receptionist', methods=['POST'])
def ai_receptionist():
    user_input = request.json.get('message', '').lower()

    if "emergency" in user_input:
        return jsonify({'response': "Please describe the emergency."})
    elif "message" in user_input or "leave a message" in user_input:
        return jsonify({'response': "Please leave your message for Dr. Adrin."})
    elif any(keyword in user_input for keyword in emergencies.keys()):
        return handle_emergency(user_input)
    else:
        return jsonify({'response': "I'm sorry, I didn't understand that. Is this an emergency or would you like to leave a message?"})

def handle_message(user_input):
    return jsonify({'response': "Thanks for the message, we will forward it to Dr. Adrin."})

def handle_emergency(user_input):
    # Simulate a delay to mimic database lookup
    time.sleep(15)

    # Find the specific emergency in the input
    for emergency, advice in emergencies.items():
        if emergency in user_input:
            return jsonify({'response': advice})

    # If no known emergency is found
    return jsonify({'response': "I'm sorry, I don't have specific advice for this situation. Please wait while Dr. Adrin is notified."})

if __name__ == '__main__':
    app.run(debug=True)


# from flask import Flask, request, jsonify, render_template
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama.llms import OllamaLLM

# app = Flask(__name__)

# # Define the prompt template
# template = """Question: {question}

# Answer: Let's think step by step."""
# prompt = ChatPromptTemplate.from_template(template)

# # Initialize the LLaMA 3 model with LangChain
# model = OllamaLLM(model="llama3")

# # Define the emergency responses manually
# emergencies = {
#     "not breathing": "Please start CPR immediately. CPR involves pushing against the chest of the patient and blowing air into their mouth in a constant rhythm.",
#     "chest pain": "Please have the patient sit down and try to relax. Avoid any physical activity until Dr. Adrin arrives.",
#     "unconscious": "Check if the patient is breathing and try to wake them up by gently shaking them or calling their name loudly."
# }

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/ai-receptionist', methods=['POST'])
# def ai_receptionist():
#     user_input = request.json.get('message', '').lower()

#     if "emergency" in user_input:
#         return jsonify({'response': "Please describe the emergency."})
#     elif "message" in user_input or "leave a message" in user_input:
#         return jsonify({'response': "Please leave your message for Dr. Adrin."})
#     elif any(keyword in user_input for keyword in emergencies.keys()):
#         return handle_emergency(user_input)
#     else:
#         # Use LangChain and LLaMA 3 for generating responses
#         response = generate_llama_response(user_input)
#         return jsonify({'response': response})

# def handle_message(user_input):
#     return jsonify({'response': "Thanks for the message, we will forward it to Dr. Adrin."})

# def handle_emergency(user_input):
#     # Simulate a delay to mimic database lookup
    
#     time.sleep(15)

#     # Find the specific emergency in the input
#     for emergency, advice in emergencies.items():
#         if emergency in user_input:
#             return jsonify({'response': advice})

#     # If no known emergency is found
#     return jsonify({'response': "I'm sorry, I don't have specific advice for this situation. Please wait while Dr. Adrin is notified."})

# def generate_llama_response(question):
#     # Use LangChain to generate a response using the LLaMA 3 model
#     chain = prompt | model
#     response = chain.invoke({"question": question})
#     return response['text']  # Extract the text from the response object

# if __name__ == '__main__':
#     app.run(debug=True)
