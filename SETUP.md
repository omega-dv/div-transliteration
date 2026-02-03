# Dhivehi Latin to Thaana Transliteration Setup Guide

This guide provides step-by-step instructions for setting up and running the **ByT5-based Dhivehi transliteration model** that converts Latin script to Thaana script.

## ⚡ Quick Start (TL;DR)

```bash
# Create environment
conda create -n div-transliteration python=3.9
conda activate div-transliteration

# Install dependencies
conda install pytorch -c pytorch
pip install transformers

# Test it (in Python)
from transformers import pipeline
translator = pipeline("text2text-generation", model="Neobe/dhivehi-byt5-latin2thaana-v1")
result = translator("salaam", max_length=256)
print(result[0]['generated_text'])
# Output: ސަލާމް
```

**Done!** Continue reading for detailed setup and usage instructions.

---

## 📊 About the Model

### **ByT5 Latin to Thaana Transliteration Model**

**Model:** [Neobe/dhivehi-byt5-latin2thaana-v1](https://huggingface.co/Neobe/dhivehi-byt5-latin2thaana-v1)

**Key Features:**
- ✅ **Byte-level T5** - Works directly on UTF-8 bytes (no tokenizer needed)
- ✅ **Specialized for Transliteration** - Optimized specifically for Latin → Thaana
- ✅ **Hybrid Approach** - Handles both formal news text and casual chat
- ✅ **Domain-Adapted** - Fine-tuned on Maldivian news media
- ✅ **Two-Stage Training** - General phonetics (150k pairs) + News domain (10k headlines)
- ✅ **Smart Grammar** - Applies formal rules in news contexts, respects casual spacing in chat

**Model Specifications:**
- **Architecture:** google/byt5-small
- **Parameters:** 0.3B
- **License:** Apache 2.0
- **Framework:** Hugging Face Transformers

**Why ByT5 for Transliteration?**

1. **Byte-level processing** - Better than subword tokenization for character-level tasks
2. **No vocabulary limitations** - Can handle any UTF-8 character
3. **Character-aware** - Naturally suited for transliteration tasks
4. **Smaller and faster** - 300M parameters, optimized for one task

### Training Details

**Two-Stage Training Pipeline:**

1. **Stage 1 - General Transliteration:**
   - Dataset: [alakxender/dhivehi-transliteration-pairs](https://huggingface.co/datasets/alakxender/dhivehi-transliteration-pairs)
   - Size: ~150,000 Latin-Thaana pairs
   - Purpose: Learn phonetics and spelling patterns

2. **Stage 2 - Domain Adaptation:**
   - Dataset: 10,000 high-quality news headlines
   - Purpose: Recognize formal entities, grammatical spacing, and journalistic phrasing

### Performance Examples

| Use Case | Latin Input | Model Output |
|----------|-------------|--------------|
| **Formal/News** | `Raeesul jumhooriyya miadhu ganoonu thasdheegu kuravvaifi` | `ރައީސުލް ޖުމްހޫރިއްޔާ މިއަދު ގަނޫނު ތަސްދީގުކުރައްވައިފި` |
| **Casual Chat** | `Aharen miadhu varah ban'du hai` | `އަހަރެން މިއަދު ވަރަށް ބަނޑުހައި` |
| **Official Titles** | `Minister of Foreign Affairs Moosa Zameer` | `މިނިސްޓަރު އޮފް ފޮރިން އެފެއާސް މޫސަ ޒަމީރު` |
| **News Phrasing** | `Police service in vanee ekan kuhveri kohfa` | `ޕޮލިސް ސާވިސްއިން ވަނީ އެކަން ކުށްވެރިކޮށްފައި` |

---

## 📋 Prerequisites

Before starting, ensure you have the following installed:

1. **Conda or Miniconda**
   - Download from: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
   - Verify installation:
     ```bash
     conda --version
     ```

2. **Git** (Optional - for cloning this repository)
   - Download from: [https://git-scm.com/downloads](https://git-scm.com/downloads)
   - Verify installation:
     ```bash
     git --version
     ```

3. **Basic System Requirements**
   - At least 2GB RAM (4GB+ recommended)
   - 1GB free disk space for dependencies and model
   - Optional: NVIDIA GPU with CUDA support for faster inference (not required)

---

## 🚀 Step-by-Step Setup Instructions

### Step 1: Create Project Directory

**Reasoning:** Organize your work in a clean project structure.

```bash
# Create and navigate to project directory
mkdir div-transliteration
cd div-transliteration
```

### Step 2: Set Up Conda Environment

**Reasoning:** Creating an isolated conda environment prevents dependency conflicts and ensures reproducibility.

```bash
# Create a new conda environment with Python 3.9
conda create -n div-transliteration python=3.9

# Activate the environment
conda activate div-transliteration
```

### Step 3: Install Required Packages

**Reasoning:** The ByT5 model uses the Hugging Face `transformers` library with PyTorch backend.

```bash
# Install PyTorch
conda install pytorch -c pytorch

# Install transformers library
pip install transformers

# Install additional useful libraries (optional)
conda install jupyter numpy pandas matplotlib
```

**For GPU Support (if you have NVIDIA GPU):**

```bash
# Install PyTorch with CUDA support instead
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Note:** GPU is optional. The model works fine on CPU for most use cases.

### Step 4: Create Transliteration Script

**Reasoning:** Create a reusable Python script to perform transliteration tasks.

Create a file called `transliterate.py`:

```python
from transformers import pipeline

# Load the model
print("Loading ByT5 model...")
translator = pipeline(
    "text2text-generation",
    model="Neobe/dhivehi-byt5-latin2thaana-v1"
)
print("Model loaded successfully!")

def latin_to_thaana(text, max_length=256):
    """
    Convert Latin script to Thaana script.

    Args:
        text: Input text in Latin script
        max_length: Maximum length of output (default: 256)

    Returns:
        Transliterated text in Thaana script
    """
    result = translator(text, max_length=max_length)
    return result[0]['generated_text']

# Example usage
if __name__ == "__main__":
    examples = [
        # Formal/News
        "Raeesul jumhooriyya miadhu ganoonu thasdheegu kuravvaifi",

        # Casual chat
        "Aharen miadhu varah ban'du hai",

        # Official title
        "Minister of Foreign Affairs Moosa Zameer",

        # Simple greeting
        "Assalaamu alaikum, kihineh haalu?",

        # News phrasing
        "Police service in vanee ekan kuhveri kohfa"
    ]

    print("\n" + "="*60)
    print("DHIVEHI LATIN TO THAANA TRANSLITERATION")
    print("="*60)

    for i, latin_text in enumerate(examples, 1):
        print(f"\n{i}. Latin:  {latin_text}")
        thaana_text = latin_to_thaana(latin_text)
        print(f"   Thaana: {thaana_text}")

    # Interactive mode
    print("\n" + "="*60)
    print("INTERACTIVE MODE (press Ctrl+C to exit)")
    print("="*60 + "\n")

    try:
        while True:
            user_input = input("Enter Latin text: ").strip()
            if user_input:
                result = latin_to_thaana(user_input)
                print(f"Thaana: {result}\n")
    except KeyboardInterrupt:
        print("\n\nExiting...")
```

### Step 5: Run the Transliteration Script

**Reasoning:** Test that the model works correctly before integrating it into your application.

```bash
# Run the script
python transliterate.py
```

**Expected output:**
```
Loading ByT5 model...
Model loaded successfully!

============================================================
DHIVEHI LATIN TO THAANA TRANSLITERATION
============================================================

1. Latin:  Raeesul jumhooriyya miadhu ganoonu thasdheegu kuravvaifi
   Thaana: ރައީސުލް ޖުމްހޫރިއްޔާ މިއަދު ގަނޫނު ތަސްދީގުކުރައްވައިފި

2. Latin:  Aharen miadhu varah ban'du hai
   Thaana: އަހަރެން މިއަދު ވަރަށް ބަނޑުހައި

3. Latin:  Minister of Foreign Affairs Moosa Zameer
   Thaana: މިނިސްޓަރު އޮފް ފޮރިން އެފެއާސް މޫސަ ޒަމީރު

4. Latin:  Assalaamu alaikum, kihineh haalu?
   Thaana: އައްސަލާމް ޢަލައިކުމް، ކިހިނެތް ހާލު؟

5. Latin:  Police service in vanee ekan kuhveri kohfa
   Thaana: ޕޮލިސް ސާވިސްއިން ވަނީ އެކަން ކުށްވެރިކޮށްފައި

============================================================
INTERACTIVE MODE (press Ctrl+C to exit)
============================================================

Enter Latin text:
```

### Step 6: Create Interactive Jupyter Notebook (Optional)

**Reasoning:** A notebook provides an interactive environment for experimenting with the model.

```bash
# Start Jupyter
jupyter notebook
```

Create a new notebook `dhivehi_transliteration.ipynb`:

```python
# Cell 1: Install and import
from transformers import pipeline

# Cell 2: Load model
translator = pipeline(
    "text2text-generation",
    model="Neobe/dhivehi-byt5-latin2thaana-v1"
)

# Cell 3: Define helper function
def transliterate(text, max_length=256):
    """Convert Latin script to Thaana script"""
    result = translator(text, max_length=max_length)
    return result[0]['generated_text']

# Cell 4: Try it out!
# Change this text and re-run the cell
latin_text = "Raeesul jumhooriyya miadhu ganoonu thasdheegu kuravvaifi"
thaana_text = transliterate(latin_text)

print(f"Latin:  {latin_text}")
print(f"Thaana: {thaana_text}")

# Cell 5: Batch processing
texts = [
    "Assalaamu alaikum",
    "Kihineh haalu?",
    "Aharen miadhu varah ban'du hai"
]

for text in texts:
    result = transliterate(text)
    print(f"{text} → {result}")
```

### Step 7: Integrate into Your Application

**Reasoning:** Use the model in your production application.

**Simple Function for Your App:**

```python
from transformers import pipeline

class DhivehiTransliterator:
    def __init__(self):
        """Initialize the transliteration model"""
        self.translator = pipeline(
            "text2text-generation",
            model="Neobe/dhivehi-byt5-latin2thaana-v1"
        )

    def latin_to_thaana(self, text, max_length=256):
        """
        Convert Latin script to Thaana script.

        Args:
            text (str): Input text in Latin script
            max_length (int): Maximum output length

        Returns:
            str: Transliterated text in Thaana script
        """
        result = self.translator(text, max_length=max_length)
        return result[0]['generated_text']

    def batch_transliterate(self, texts, max_length=256):
        """
        Transliterate multiple texts at once.

        Args:
            texts (list): List of Latin script texts
            max_length (int): Maximum output length

        Returns:
            list: List of transliterated Thaana texts
        """
        results = self.translator(texts, max_length=max_length)
        return [r['generated_text'] for r in results]

# Usage
transliterator = DhivehiTransliterator()

# Single text
thaana = transliterator.latin_to_thaana("Assalaamu alaikum")
print(thaana)

# Batch processing (more efficient)
texts = ["Hello", "How are you?", "Good morning"]
results = transliterator.batch_transliterate(texts)
for latin, thaana in zip(texts, results):
    print(f"{latin} → {thaana}")
```

**Web API Example (Flask):**

```python
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load model once at startup
translator = pipeline(
    "text2text-generation",
    model="Neobe/dhivehi-byt5-latin2thaana-v1"
)

@app.route('/transliterate', methods=['POST'])
def transliterate():
    """API endpoint for transliteration"""
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    result = translator(text, max_length=256)
    thaana = result[0]['generated_text']

    return jsonify({
        'latin': text,
        'thaana': thaana
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

Test the API:
```bash
curl -X POST http://localhost:5000/transliterate \
  -H "Content-Type: application/json" \
  -d '{"text": "Assalaamu alaikum"}'
```

---

## 🎯 Usage Tips

### Best Practices

1. **Batch Processing** - Process multiple texts together for better performance
   ```python
   texts = ["text1", "text2", "text3"]
   results = translator(texts, max_length=256)
   ```

2. **Adjust max_length** - Set based on your expected output length
   - Short messages: `max_length=128`
   - Normal text: `max_length=256` (default)
   - Long articles: `max_length=512`

3. **Model Caching** - Load the model once and reuse
   ```python
   # Good - load once
   translator = pipeline("text2text-generation", model="Neobe/dhivehi-byt5-latin2thaana-v1")

   # Bad - loads model every time
   def transliterate(text):
       translator = pipeline(...)  # Don't do this!
   ```

### Use Cases

**Best for:**
- ✅ News article transliteration
- ✅ Official documents
- ✅ Social media posts
- ✅ Chat messages
- ✅ Mixed formal/casual text
- ✅ Proper nouns and titles

**Limitations:**
- ❌ Only supports Latin → Thaana (not bidirectional)
- ❌ Not for translation (only transliteration)
- ❌ Optimized for Maldivian news domain

---

## 🐍 Conda Environment Management

### Useful Conda Commands

```bash
# List all conda environments
conda env list

# Activate environment
conda activate div-transliteration

# Deactivate environment
conda deactivate

# List installed packages in current environment
conda list

# Export environment to YAML file
conda env export > environment.yml

# Remove environment (if needed)
conda env remove -n div-transliteration

# Update conda itself
conda update conda

# Update all packages in environment
conda update --all
```

### Creating an Environment File

For better reproducibility, create an `environment.yml` file:

```yaml
name: div-transliteration
channels:
  - defaults
  - conda-forge
  - pytorch
dependencies:
  - python=3.9
  - pytorch
  - numpy
  - pandas
  - matplotlib
  - jupyter
  - pip
  - pip:
    - transformers>=4.30.0
```

Save this to `environment.yml` in your project directory, then create the environment with:

```bash
conda env create -f environment.yml
```

---

## 🔧 Troubleshooting

### Common Issues

1. **Missing Dependencies:** Ensure all packages are installed with correct versions
   ```bash
   conda list  # Check installed packages
   conda install <package-name>  # Install missing packages
   ```

2. **Dependency Conflicts (e.g., sympy version):**
   ```bash
   # Check what PyTorch needs
   pip show torch

   # Install the specific version
   pip install sympy==1.13.1
   ```

3. **GPU Issues:** If you have GPU but it's not being used
   ```bash
   # Check if CUDA is available
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

   # Reinstall PyTorch with CUDA
   conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

4. **Conda Environment Not Found:** Make sure to activate the environment
   ```bash
   conda activate div-transliteration
   ```

5. **Model Download Issues:** If the model fails to download
   ```bash
   # Check your internet connection
   # Try manually downloading from: https://huggingface.co/Neobe/dhivehi-byt5-latin2thaana-v1

   # Or set HuggingFace cache directory
   export HF_HOME=/path/to/cache
   ```

6. **Memory Issues:** If you get out-of-memory errors
   ```python
   # Use smaller batch sizes
   result = translator(text, max_length=128)  # Reduce max_length

   # Or process one at a time instead of batches
   ```

7. **Slow Performance:** To speed up inference
   ```python
   # Use GPU if available
   import torch
   device = 0 if torch.cuda.is_available() else -1
   translator = pipeline("text2text-generation", model="...", device=device)

   # Or reduce max_length
   result = translator(text, max_length=128)
   ```

8. **Conda is Slow:** Use mamba for faster package resolution
   ```bash
   conda install mamba -c conda-forge
   mamba install pytorch  # Use mamba instead of conda
   ```

### Performance Tips

1. **Use GPU acceleration** if available for faster inference (3-5x speedup)
2. **Batch processing** - Process multiple texts together
3. **Reduce max_length** if you know outputs will be short
4. **Cache the model** - Load once, use many times
5. **Use smaller batches** if experiencing memory issues

---

## 📚 Additional Resources

### Model Resources

- **[Model Card on Hugging Face](https://huggingface.co/Neobe/dhivehi-byt5-latin2thaana-v1)** - Official model page
- **[ByT5 Paper](https://arxiv.org/abs/2105.13626)** - "ByT5: Towards a token-free future with pre-trained byte-to-byte models"
- **[Hugging Face Transformers Docs](https://huggingface.co/docs/transformers/)** - Official documentation

### Datasets

- **[alakxender/dhivehi-transliteration-pairs](https://huggingface.co/datasets/alakxender/dhivehi-transliteration-pairs)** - Main training dataset (~150k pairs)

### Dhivehi Language Resources

- **[dhivehi.ai](https://dhivehi.ai/docs/)** - Comprehensive documentation of Dhivehi NLP resources
- **[Thaana Script Information](https://en.wikipedia.org/wiki/Thaana)** - Background on the Thaana writing system
- **[Dhivehi Writing Systems](https://r12a.github.io/scripts/thaa/dv.html)** - Orthography and transliteration details
- **[Unicode CLDR Latin-Thaana Chart](https://www.unicode.org/cldr/cldr-aux/charts/30/transforms/Latin-Thaana.html)** - Standard transliteration mappings

### Python & Machine Learning

- **[PyTorch Documentation](https://pytorch.org/docs/stable/index.html)** - PyTorch official docs
- **[Conda User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/)** - Conda environment management
- **[Hugging Face Course](https://huggingface.co/course)** - Free course on transformers and NLP

---

## 📝 Next Steps

### Immediate Actions (Recommended)

1. **✅ Set up conda environment** (5 minutes)
   ```bash
   conda create -n div-transliteration python=3.9
   conda activate div-transliteration
   ```

2. **✅ Install dependencies** (5-10 minutes)
   ```bash
   conda install pytorch -c pytorch
   pip install transformers
   ```

3. **✅ Create and run the transliteration script** (10 minutes)
   - Copy the `transliterate.py` code from Step 4
   - Run it and verify the model works

4. **✅ Integrate into your application** (varies)
   - Use the `DhivehiTransliterator` class in your code
   - Build a web API (Flask/FastAPI)
   - Create a CLI tool
   - Deploy to cloud (AWS/GCP/Azure)

### Optional Actions (Based on Your Needs)

5. **🔬 Fine-tune the model** (if you have custom data)
   - Prepare your domain-specific Latin-Thaana parallel corpus
   - Fine-tune using Hugging Face Trainer API
   - Evaluate on your test set

6. **📊 Explore the dataset** (if you need training data)
   - Visit [alakxender/dhivehi-transliteration-pairs](https://huggingface.co/datasets/alakxender/dhivehi-transliteration-pairs)
   - Download samples to understand the format
   - Use for additional fine-tuning if needed

7. **🚀 Deploy to production**
   - Containerize with Docker
   - Deploy to cloud platforms
   - Set up API endpoints
   - Add monitoring and logging

### Getting Help

- **Issues with setup?** Check the Troubleshooting section above
- **Questions about the model?** Visit the [Hugging Face model page](https://huggingface.co/Neobe/dhivehi-byt5-latin2thaana-v1)
- **Need more Dhivehi NLP resources?** Browse [dhivehi.ai](https://dhivehi.ai/docs/)

---

## 📖 References

- [Neobe/dhivehi-byt5-latin2thaana-v1 on Hugging Face](https://huggingface.co/Neobe/dhivehi-byt5-latin2thaana-v1)
- [alakxender/dhivehi-transliteration-pairs Dataset](https://huggingface.co/datasets/alakxender/dhivehi-transliteration-pairs)
- [ByT5 Paper - Towards a token-free future](https://arxiv.org/abs/2105.13626)
- [dhivehi.ai - Dhivehi NLP Resources](https://dhivehi.ai/docs/)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)

---

**Last Updated:** February 3, 2026
**Model:** Neobe/dhivehi-byt5-latin2thaana-v1
**Environment Manager:** Conda/Miniconda
