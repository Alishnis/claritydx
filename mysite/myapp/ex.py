import os
import re
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, logging as hf_logging
from deep_translator import GoogleTranslator

MODEL_NAME = "microsoft/BioGPT"
MODEL_LOAD_ERROR = (
    "Symptom analysis is temporarily unavailable. "
    "Please try again later or use the Large Database section."
)
NO_RESULT_ERROR = (
    "Unable to generate a diagnosis right now. "
    "Please try rephrasing the symptoms."
)


def fail(message, exit_code=1):
    print(message, file=sys.stderr)
    raise SystemExit(exit_code)


def main():
    hf_logging.set_verbosity_error()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    symptoms = input().strip()

    if not symptoms:
        fail("No symptoms provided.")

    try:
        # Translate to English when needed
        try:
            symptoms_en = GoogleTranslator(source='auto', target='en').translate(symptoms)
        except Exception:
            # Use the original text if translation fails instead of breaking the flow.
            symptoms_en = symptoms

        print("Loading the model and tokenizer... This might take a few seconds.", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

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

        if not diagnosis_en:
            fail(NO_RESULT_ERROR)

        print(diagnosis_en)

    except SystemExit:
        raise
    except Exception:
        fail(MODEL_LOAD_ERROR)

if __name__ == "__main__":
    main()
