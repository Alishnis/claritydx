import re
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from deep_translator import GoogleTranslator

def main():
    symptoms = input().strip()

    if not symptoms:
        print("No symptoms provided.", file=sys.stderr)
        return

    try:
        # Translate to English when needed
        symptoms_en = GoogleTranslator(source='auto', target='en').translate(symptoms)

        print("Loading the model and tokenizer... This might take a few seconds.", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")
        model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT")

        print("Analyzing symptoms...", file=sys.stderr)
        input_text = f"Patient symptoms: {symptoms_en}. Based on these symptoms, provide possible diagnoses:"
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=100,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            num_return_sequences=1
        )

        diagnosis_en = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Try to keep only the answer portion
        marker = "possible diagnoses:"
        lower = diagnosis_en.lower()
        if marker in lower:
            diagnosis_en = diagnosis_en[lower.find(marker) + len(marker):]
        diagnosis_en = re.sub(r"\s+", " ", diagnosis_en).strip()

        print(diagnosis_en)

    except Exception as e:
        print(f"An error occurred: {str(e)}", file=sys.stderr)

if __name__ == "__main__":
    main()
