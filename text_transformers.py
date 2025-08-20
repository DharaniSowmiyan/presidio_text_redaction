import spacy
import os
from presidio_analyzer import AnalyzerEngine, RecognizerResult, RecognizerRegistry, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngine, NlpArtifacts
from presidio_anonymizer import AnonymizerEngine
from typing import List, Any

class PreloadedSpacyNlpEngine(NlpEngine):
    """
    A custom NlpEngine that accepts a pre-loaded spaCy nlp object.
    """
    def __init__(self, spacy_model: Any):
        super().__init__()
        self.nlp = {"en": spacy_model}

    def load(self):
        """The model is already loaded, so this is a no-op."""
        pass

    def is_loaded(self) -> bool:
        """Always returns True as the model is loaded during init."""
        return self.nlp is not None

    def process_text(self, text: str, language: str) -> NlpArtifacts:
        """
        Processes the text and returns an NlpArtifacts object.
        """
        doc = self.nlp[language](text)
        return NlpArtifacts(
            nlp_engine=self,
            entities=list(doc.ents),
            tokens=[token.text for token in doc],
            tokens_indices=[token.idx for token in doc],
            lemmas=[token.lemma_ for token in doc],
            language=language
        )

    def get_supported_languages(self, **kwargs) -> List[str]:
        return list(self.nlp.keys())

    def get_supported_entities(self, **kwargs) -> List[str]:
        return []

    def is_stopword(self, word: str, language: str) -> bool:
        return self.nlp[language].vocab[word].is_stop

    def is_punct(self, word: str, language: str) -> bool:
        return self.nlp[language].vocab[word].is_punct

    def process_batch(self, texts: List[str], language: str, **kwargs) -> List[Any]:
        docs = self.nlp[language].pipe(texts, **kwargs)
        nlp_artifacts_list = []
        for i, doc in enumerate(docs):
            nlp_artifacts_list.append(NlpArtifacts(
                nlp_engine=self,
                entities=list(doc.ents),
                tokens=[token.text for token in doc],
                tokens_indices=[token.idx for token in doc],
                lemmas=[token.lemma_ for token in doc],
                language=language
            ))
        return nlp_artifacts_list

def run_redaction(input_path, output_path):
    """
    Main function to run the PII redaction process on a file.
    """
    # --- 1. Read the input file ---
    print(f"Reading text from '{input_path}'...")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            text_to_redact = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # --- 2. Build the custom spaCy pipeline ---
    print("Initializing NLP models (this may take a moment)...")
    hf_pipeline_config = {
        "model": "dslim/bert-base-NER",
        "kwargs": {"framework": "pt"}
    }
    nlp = spacy.load("en_core_web_sm")
    if "hf_token_pipe" not in nlp.pipe_names:
        nlp.add_pipe("hf_token_pipe", config=hf_pipeline_config)

    # --- 3. Set up the Analyzer with our Custom Engine ---
    nlp_engine = PreloadedSpacyNlpEngine(spacy_model=nlp)
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["en"]
    )

    # --- 4. Analyze the text for PII ---
    print("Analyzing text for PII...")
    analyzer_results = analyzer.analyze(
        text=text_to_redact,
        language='en'
    )

    # --- 5. Anonymize the text ---
    anonymizer = AnonymizerEngine()
    anonymized_result = anonymizer.anonymize(
        text=text_to_redact,
        analyzer_results=analyzer_results
    )

    # --- 6. Write the output file ---
    print(f"Writing anonymized text to '{output_path}'...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(anonymized_result.text)
    except Exception as e:
        print(f"Error writing file: {e}")
        return

    # --- 7. Print summary ---
    print("\n" + "="*50)
    print("Redaction Complete!")
    print(f"Output saved to: {output_path}")
    print("="*50)


if __name__ == "__main__":
    # Interactively ask the user for the input file path
    input_file = input("Please enter the path to the input text file: ")

    # Check if the file exists before proceeding
    if os.path.exists(input_file):
        # Automatically generate the output file path
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_transformer_output{ext}"
        
        # Run the main redaction process
        run_redaction(input_file, output_file)
    else:
        print(f"Error: Input file not found at '{input_file}'")
