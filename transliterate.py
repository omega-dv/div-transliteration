from transformers import pipeline

# Load the model
print("Loading ByT5 model...")
translator = pipeline(
    "text2text-generation",
    model="Neobe/dhivehi-byt5-latin2thaana-v1"
)
print("Model loaded successfully!")

def latin_to_thaana(text, max_new_tokens=None):
    """
    Convert Latin script to Thaana script.

    Args:
        text: Input text in Latin script
        max_new_tokens: Maximum length of output (default: auto-calculated)

    Returns:
        Transliterated text in Thaana script
    """
    # Auto-calculate max_new_tokens based on input length if not specified
    if max_new_tokens is None:
        estimated_tokens = len(text.split()) * 2  # Rough estimate
        max_new_tokens = max(512, min(estimated_tokens, 1024))  # Between 512-1024

    result = translator(text, max_new_tokens=max_new_tokens)
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
