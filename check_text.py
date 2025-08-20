import spacy
import os
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine

def main(input_path, output_path):
    """
    Analyzes a text file for PII, redacts it, and saves the output.

    :param input_path: Path to the source text file.
    :param output_path: Path to save the anonymized text file.
    """
    # --- 1. Set up the Analyzer and Anonymizer ---
    # You can customize the engine with custom recognizers or models here
    # For this example, we will use the default AnalyzerEngine
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    # --- 2. Read text from the input file ---
    print(f"Reading text from '{input_path}'...")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            text_to_redact = f.read()
    except FileNotFoundError:
        # This is a fallback, but the main block should catch this first.
        print(f"Error: Input file not found at '{input_path}'")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # --- 3. Analyze the text for PII ---
    print("Analyzing text for PII...")
    analyzer_results = analyzer.analyze(text=text_to_redact, language='en')

    # --- 4. Anonymize the text ---
    print("Anonymizing PII...")
    anonymized_result = anonymizer.anonymize(
        text=text_to_redact,
        analyzer_results=analyzer_results
    )

    # --- 5. Write the anonymized text to the output file ---
    print(f"Writing anonymized text to '{output_path}'...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(anonymized_result.text)
    except Exception as e:
        print(f"An error occurred while writing the file: {e}")
        return

    # --- 6. Print a summary to the console ---
    print("\n" + "="*50)
    print("Redaction Complete!")
    print(f"Original text read from: {input_path}")
    print(f"Anonymized text saved to: {output_path}")
    print("\nRedacted Items and Replacements:")
    if anonymized_result.items:
        for item in anonymized_result.items:
            # Note: The 'item.text' in the anonymizer result is the replacement text (e.g., <PERSON>)
            # To show the original text, we slice the input string.
            original_value = text_to_redact[item.start:item.end]
            print(f"  - Original: '{original_value}', Replaced with: '{item.entity_type}'")
    else:
        print("  No PII was found to redact.")
    print("="*50)


if __name__ == "__main__":
    # Interactively ask the user for the input file path
    input_file = input("Please enter the path to the input text file: ")

    # Check if the file exists before proceeding
    if os.path.exists(input_file):
        # Automatically generate the output file path
        # e.g., 'my_document.txt' becomes 'my_document_output.txt'
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_output{ext}"
        
        # Run the main function with the generated paths
        main(input_file, output_file)
    else:
        print(f"Error: Input file not found at '{input_file}'")
