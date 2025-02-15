from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from retrieve_result import retreival_result, result_after_retreival
from langchain_community.vectorstores import Pinecone as LangchainPinecone  # Corrected import

from dotenv import load_dotenv
import os

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Fetch the API keys from environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# Debugging: Print the keys to check if they are loaded correctly
print("PINECONE_API_KEY:", PINECONE_API_KEY)
print("GROQ_API_KEY:", GROQ_API_KEY)

# Ensure the environment variables are set
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in the .env file.")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file.")

# Set the API keys as environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = GROQ_API_KEY  # Assuming you meant to use GROQ_API_KEY here

# Initialize embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone index
index_name = "medical-vector"
docsearch = LangchainPinecone.from_existing_index(index_name, embeddings)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input_message = msg.lower()  # Convert to lowercase for easier matching
    print(f"User Input: {input_message}")

    # Retrieve documents based on the input message
    docs = retreival_result(PINECONE_API_KEY, input_message, docsearch)
    
    # Generate response after retrieval
    response = result_after_retreival(GROQ_API_KEY, input_message, docs)

    # Concatenate the response into a single string
    full_response = ''.join(response)
    print(f"Response: {full_response}")

    # General responses based on user input
    if any(greeting in input_message for greeting in ["hi", "hello", "hey"]):
        return "Hello! How can I assist you today?"
    elif any(farewell in input_message for farewell in ["bye", "goodbye"]):
        return "Goodbye! Take care."
    elif "thanks" in input_message or "thank you" in input_message:
        return "You're welcome! Let me know if you have any other questions."
    elif full_response:  # If there's a relevant response
        return full_response
    else:  # Fallback response for unclear queries
        return "I'm sorry, I'm not sure about that."

if __name__ == '__main__':
    # Run the Flask app on port 8081
    app.run(host="0.0.0.0", port=8081, debug=True)
