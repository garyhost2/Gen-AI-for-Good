from flask import Flask
from flask import Blueprint, render_template, session, flash, request, redirect, url_for, jsonify
import os

from llama_cpp import Llama

app = Flask(__name__)

#MODEL_PATH = r"model\models--Hamatoysin--autibot\snapshots\ba8cbad4719a198ea5bb1d8dff51bc2c5afd6138\unsloth.Q8_0.gguf"



#client = Llama(model_path=MODEL_PATH) this is for local model after download

# this to download the model
client = Llama.from_pretrained(
	repo_id="Hamatoysin/autibot",
	filename="unsloth.Q8_0.gguf",
)

@app.route('/')
def index():
    return render_template('siderbar.html')

@app.route('/Music')
def Music():
    return render_template('Music.html')

@app.route('/quiz')
def quiz():
    return render_template('quiz.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')
@app.route('/MemoryGame')
def MemoryGame():
    return render_template('MemoryGame.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/GameSelectTest')
def GameSelectTest():
    return render_template('GameSelectTest.html')

@app.route('/chess')
def chess():
    return render_template('chess.html')



@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.json
    prompt = data['prompt']
    response = get_llm_response(prompt)
    return jsonify({'response': response})

def get_llm_response(prompt):
    # Initialize an empty string to accumulate the response
    response = ""
    
    # Create the chat completion with streaming enabled
    stream = client.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are Finn, a dedicated and understanding autism assistance bot. Your role is to support individuals on the autism spectrum by providing clear, respectful, and informative guidance. Ensure your communication is structured, predictable, and free from ambiguous language. Please avoid using slang, overly casual expressions, or phrases that may be confusing, such as 'LOL' or 'Hang tight.' Focus on offering practical advice, emotional support, and resources tailored to the unique needs of each user. Always prioritize empathy, patience, and clarity in your interactions."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=True,
    )
    
    # Iterate over the streamed chunks and accumulate the content
    for chunk in stream:
        choices = chunk.get('choices', [])
        if choices:
            delta = choices[0].get('delta', {})
            delta_content = delta.get('content')
            if delta_content:
                response += delta_content
    
    return response





if __name__ == '__main__':
    app.run()
