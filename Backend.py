from flask import Flask, jsonify, request, render_template, redirect, url_for
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Initialize Flask app
app = Flask(__name__)

# Load the default model and tokenizer
model_name = "microsoft/DialoGPT-medium"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to get a smart response from the model
def chat_with_model(input_text):
    # Add a more specific prompt for smarter responses
    prompt = f"User asked: {input_text}. Please provide a detailed and informative explanation."

    # Generate the response
    new_user_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
    chat_history_ids = model.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=3, temperature=0.7)
    bot_output = tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return bot_output

# Home route to redirect from root URL
@app.route('/')
def home_redirect():
    return redirect(url_for('home'))

# Home page (home.html now)
@app.route('/home')
def home():
    return render_template('home.html')

# Chat page
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_input = request.form['user_input']
        bot_response = chat_with_model(user_input)
        return render_template('chat.html', bot_response=bot_response)
    return render_template('chat.html')

# API creation page (for people to create their own AI APIs)
@app.route('/apis', methods=['GET', 'POST'])
def apis():
    if request.method == 'POST':
        # Process form data to create/save an API (just an example placeholder)
        api_name = request.form['api_name']
        description = request.form['description']
        
        # Logic to save the API can go here (e.g., save to a database or file)
        
        return render_template('apis.html', success=True, api_name=api_name)
    
    return render_template('apis.html', success=False)

# Route for favicon (Optional, to stop the 404 error for favicon.ico)
@app.route('/favicon.ico')
def favicon():
    return '', 204  # Empty response with a 204 status (No Content)

if __name__ == '__main__':
    app.run(debug=True)
