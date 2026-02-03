from flask import Flask, render_template, request, jsonify, Response
from transformers import pipeline, TextIteratorStreamer, AutoTokenizer, AutoModelForSeq2SeqLM
import time
import json
import threading

app = Flask(__name__)

# Load model and tokenizer for streaming support
print("Loading ByT5 model...")
model_name = "Neobe/dhivehi-byt5-latin2thaana-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print("Model loaded successfully!")

# Store active generations
active_generations = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transliterate', methods=['POST'])
def transliterate():
    """API endpoint - uses working pipeline method"""
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Generate request ID
    request_id = str(time.time())

    def generate():
        try:
            # Send initial status
            yield f"data: {json.dumps({'status': 'Generating...', 'request_id': request_id})}\n\n"

            # Store that this generation is active
            active_generations[request_id] = True

            # Create streamer for token-by-token output
            streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt")

            # Generation kwargs
            generation_kwargs = dict(
                inputs,
                max_new_tokens=4096,
                streamer=streamer,
            )

            # Run generation in background thread
            def do_transliteration():
                if request_id in active_generations:
                    model.generate(**generation_kwargs)

            thread = threading.Thread(target=do_transliteration)
            thread.start()

            # Stream tokens as they're generated
            generated_text = ""
            for new_text in streamer:
                if request_id not in active_generations:
                    # Generation was stopped
                    yield f"data: {json.dumps({'status': 'Stopped', 'thaana': generated_text, 'partial': True})}\n\n"
                    return

                generated_text += new_text
                # Send partial result
                yield f"data: {json.dumps({'status': 'Generating...', 'thaana': generated_text, 'partial': True})}\n\n"

            # Wait for thread to complete
            thread.join()

            # Send final result
            yield f"data: {json.dumps({'status': 'Complete!', 'thaana': generated_text, 'latin': text, 'partial': False})}\n\n"

            # Cleanup
            if request_id in active_generations:
                del active_generations[request_id]

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            if request_id in active_generations:
                del active_generations[request_id]

    return Response(generate(), mimetype='text/event-stream')

@app.route('/stop/<request_id>', methods=['POST'])
def stop_generation(request_id):
    """Stop an active generation"""
    if request_id in active_generations:
        del active_generations[request_id]
        return jsonify({'status': 'stopped'})
    return jsonify({'status': 'not_found'}), 404

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 Dhivehi Transliteration Web App")
    print("="*60)
    print("📍 Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)
