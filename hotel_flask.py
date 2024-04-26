from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import spacy
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json
from queue import Queue
import threading
import asyncio

# Load vocabulary from JSON file
with open('vocab.json', 'r') as f:
    vocab = json.load(f)

# spacy.cli.download("en_core_web_lg")
nlp = spacy.load("en_core_web_sm")

class HybridHotelModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, bidirectional, pretrained_embeddings=None):
        super(HybridHotelModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(pretrained_embeddings)
            self.embedding.weight.requires_grad = True  # Allow embeddings to be trainable
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=0.5 if num_layers > 1 else 0)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2 if bidirectional else hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Softmax activation for multi-class classification

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_output, (lstm_hidden, _) = self.lstm(embedded)
        
        # Concatenate forward and backward hidden states if bidirectional
        if lstm_hidden.shape[0] > 1:
            lstm_hidden = torch.cat((lstm_hidden[-2,:,:], lstm_hidden[-1,:,:]), dim=1)
        else:
            lstm_hidden = lstm_hidden[-1,:,:]

        lstm_hidden = self.dropout(lstm_hidden)
        lstm_hidden = self.batch_norm(lstm_hidden)
        out = self.fc(lstm_hidden)
        out = self.softmax(out)  # Apply softmax activation to get class probabilities
        return out
    
vocab_size = len(vocab)
embedding_dim = 250
hidden_dim = 256
output_dim = 6  # Adjust this according to the number of classes in your multi-class classification task
num_layers = 2
bidirectional = True
pretrained_embeddings = None 
max_seq_length = 100 # You can load pretrained embeddings here if desired

# Initializing the hybrid model with specified hyperparameters
hybrid_model = HybridHotelModel(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, bidirectional, pretrained_embeddings)

hybrid_model.load_state_dict(torch.load('hotelhybrid_model.pth'))

# Set the model to evaluation mode
hybrid_model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = pd.read_csv("hotels_10000.csv", encoding='latin-1')

label_encoder = LabelEncoder()

data[' HotelRating'] = label_encoder.fit_transform(data[' HotelRating'])

rating_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
print("Encoded ratings mapping:", rating_mapping)

custom_mapping = {'All': 0, 'OneStar': 1, 'TwoStar': 2, 'ThreeStar': 3, 'FourStar': 4, 'FiveStar': 5}

data[' HotelRating'] = data[' HotelRating'].replace(custom_mapping)

reverse_mapping = {value: key for key, value in custom_mapping.items()}

# Define function to tokenize new text
def tokenize_new_text(new_text):
    tokens = [token.text for token in nlp(new_text.lower()) if not token.is_stop and not token.is_punct]
    return tokens

# Initialize Flask app
app = Flask(__name__)

# Define route for home page
@app.route('/')
async def home():
    return await asyncio.to_thread(render_template, 'indexes.html')

# Define a function to perform prediction asynchronously
def predict_async(model, new_text, vocab, reverse_mapping, device, result_queue):
    # Tokenize the new text
    new_text_tokens = tokenize_new_text(new_text)

    # Map tokens to numerical indices based on the vocabulary
    numerical_sequence = [vocab[token] for token in new_text_tokens]

    # Pad or truncate the numerical sequence
    padded_sequence = numerical_sequence + [0] * (max_seq_length - len(numerical_sequence))
    padded_sequence_tensor = torch.tensor([padded_sequence]).to(device)

    # Pass the tensor through the trained model to get the prediction
    with torch.no_grad():
        output = model(padded_sequence_tensor)

    # Get the predicted class (index with highest probability)
    _, predicted_class = torch.max(output, 1)

    # Decode the predicted class to obtain the rating label
    predicted_rating = reverse_mapping[predicted_class.item()]

    # Put the result in the queue
    result_queue.put(predicted_rating)

# Define route for prediction
@app.route('/predict', methods=['POST'])
async def predict():
    # Get new text from request data
    request_data = request.get_json()
    if request_data is None or 'text' not in request_data:
        return jsonify({"error": "Invalid request data"}), 400
    new_text = request_data['text']

    # Create a queue to pass the result back from the prediction thread
    result_queue = Queue()

    # Perform prediction asynchronously
    prediction_thread = threading.Thread(target=predict_async, args=(hybrid_model, new_text, vocab, reverse_mapping, device, result_queue))
    prediction_thread.start()
    prediction_thread.join(timeout=10)  # Wait for prediction with timeout

    # Check if prediction thread has finished
    if prediction_thread.is_alive():
        return jsonify({"error": "Prediction timed out"}), 500
    else:
        predicted_rating = result_queue.get()
        return jsonify({"response": f'Based on your requests the best rated hotels will start from this {predicted_rating} rating and the price will be negotiable as for your request'})
    
# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
