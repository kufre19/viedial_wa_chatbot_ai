import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import utils

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Default document path
DEFAULT_FOLDER = "documents"

@app.route('/', methods=['GET'])
def indexTest():
    # render the html file 
    return render_template('index.html')


@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize the embedding database with all documents in the folder."""
    try:
        documents_dir = os.path.join(os.path.dirname(__file__), DEFAULT_FOLDER)
        result = utils.process_folder_and_create_embeddings(documents_dir)
        return jsonify(result)
    except Exception as e:
        # Log the error to a file
        with open('error.log', 'a') as f:
            f.write(f"Error during initialization: {str(e)}\n")
        return jsonify({"success": False, "error": str(e)}), 500
    
    
@app.route('/api/auto_generate_chat_title', methods=['POST'])
def auto_generate_chat_title():
    """Auto generate a chat title based on the question."""
    data = request.get_json()
    question = data['question']
    result = utils.auto_generate_chat_title(question)
    return jsonify(result)



@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Answer a question based on the document."""
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({"success": False, "error": "Question is required"}), 400
    
    question = data['question']
    
    
    # Get answer
    result = utils.get_answer_for_question(question)
    return jsonify(result)



@app.route('/api/test', methods=['POST'])
def test():
    """Test the search functionality."""
    data = request.get_json()
    question = data['question']
    result = utils.test_search(question)
    return jsonify(result)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000) 