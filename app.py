from flask import Flask, render_template, request, jsonify, Response
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
import json
import re

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
            yield f"data: {json.dumps({'status': 'Starting...', 'request_id': request_id})}\n\n"

            # Store that this generation is active
            active_generations[request_id] = True

            # Split text into sentences while preserving punctuation
            # This pattern captures sentences WITH their ending punctuation
            sentence_pattern = r'[^.!?]+[.!?]+|[^.!?]+$'
            sentences = re.findall(sentence_pattern, text)
            sentences = [s.strip() for s in sentences if s.strip()]

            # If no sentences found, treat entire text as one sentence
            if not sentences:
                sentences = [text]

            # Process each sentence with beam search
            all_thaana = []
            total_sentences = len(sentences)

            for idx, sentence in enumerate(sentences, 1):
                if request_id not in active_generations:
                    # Generation was stopped
                    yield f"data: {json.dumps({'status': 'Stopped', 'thaana': ' '.join(all_thaana), 'partial': True})}\n\n"
                    return

                # Update status
                status_msg = f'Processing sentence {idx}/{total_sentences}...'
                yield f"data: {json.dumps({'status': status_msg, 'request_id': request_id})}\n\n"

                # Extract ending punctuation to preserve it
                ending_punct = ''
                sentence_text = sentence
                if sentence and sentence[-1] in '.!?':
                    ending_punct = sentence[-1]
                    sentence_text = sentence[:-1].strip()

                # Tokenize sentence (without the punctuation)
                inputs = tokenizer(sentence_text, return_tensors="pt", truncation=False, padding=False)

                # Generate with beam search for quality
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    num_beams=4,
                    do_sample=False,
                    early_stopping=False,
                    length_penalty=1.2,
                )

                # Decode the output and add back the punctuation
                sentence_thaana = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if ending_punct:
                    sentence_thaana += ending_punct
                all_thaana.append(sentence_thaana)

                # Send partial result with completed sentences
                partial_result = ' '.join(all_thaana)
                yield f"data: {json.dumps({'status': status_msg, 'thaana': partial_result, 'partial': True})}\n\n"

            # Send final result
            final_thaana = ' '.join(all_thaana)
            yield f"data: {json.dumps({'status': 'Complete!', 'thaana': final_thaana, 'latin': text, 'partial': False})}\n\n"

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
