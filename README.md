
# Medical Report Generator and Translator

### A comprehensive tool for creating, translating, and evaluating the translation of medical reports across different languages.

Generate synthetic medical reports (or ingest existing `.txt` files), translate them into multiple languages using any LLM, and then evaluate the quality of the translations using various metrics.

### Usage
```bash
python medical_report_generator_and_translator.py [options]
python medical_report_generator_and_translator.py --languages "Spanish,Portuguese,Haitian Creole,Vietnamese" 
```

**Output:**  
Creates a new folder with translations for each specified language in `.csv`, `.txt`, and `.json` formats, along with a corresponding evaluation report.

### Options
- `--languages TEXT` &nbsp;&nbsp;&nbsp;&nbsp; Comma-separated list of target languages for translation
- `--model TEXT` &nbsp;&nbsp;&nbsp;&nbsp; LLM model to use for translation (default: `gpt-4o-mini`)
- `--guidance TEXT` &nbsp;&nbsp;&nbsp;&nbsp; Additional specifications for synthetic report generation, such as socio-demographics or disease details
- `--input_file TEXT` &nbsp;&nbsp;&nbsp;&nbsp; Path to a `.txt` file with a medical report (to use instead of a synthetic report)
- `--api_key TEXT` &nbsp;&nbsp;&nbsp;&nbsp; OpenAI API key (required if not present in `.env` file)

> **Note:** Ensure that you have `OPENAI_API_KEY` in the `.env` file before running the script, or provide it via the `--api_key` argument.

### Translation Evaluation Metrics

1. **BLEU (Bilingual Evaluation Understudy):**
   - Measures the precision of n-gram matches between the translated text and reference text.
   - Important for assessing overall translation quality.

2. **METEOR (Metric for Evaluation of Translation with Explicit ORdering):**
   - Considers synonyms and paraphrases, which is particularly useful for medical terminology.
   - Provides a more nuanced evaluation of semantic similarity.

3. **RIBES (Rank-based Intuitive Bilingual Evaluation Score):**
   - Evaluates word order in translations, crucial for maintaining the logical flow of medical information.

4. **BERTScore:**
   - Uses contextual embeddings to compute similarity, capturing semantic meaning better than exact matches.
   - Particularly useful for assessing the preservation of medical concepts.

5. **Readability Metrics (Flesch Reading Ease):**
   - Ensures that the translated text maintains an appropriate level of readability, crucial for patient understanding in medical contexts.

6. **Terminology Accuracy:**
   - Checks the accuracy of key medical terms in the translation, ensuring that critical medical information is preserved.

7. **Consistency Check:**
   - Evaluates how consistently specific terms or phrases are translated throughout the document, essential for maintaining clarity in medical reports.

8. **Error Categorization:**
   - Categorizes errors (e.g., terminology, grammar, style) to provide more actionable feedback for improving translations.

9. **Cultural Appropriateness Assessment:**
   - Evaluates the cultural sensitivity and appropriateness of the translations, crucial in healthcare communication across different cultures.

---

# Installation and Usage Guide

This guide provides instructions on how to install the required packages and run the Medical Report Generator and Translator.

### 1. Installation

Ensure you have Python 3.7+ installed on your system. Then, install the required packages using pip:

```bash
pip install -r requirements.txt
```

### 2. Usage

To run the script, use the following command format:

```bash
python medical_report_generator_and_translator.py [options]
```

#### Examples:

a) Generate a synthetic report and translate it to default languages:

```bash
python medical_report_generator_and_translator.py --api_key your_openai_api_key
```

b) Translate an existing report to specific languages:

```bash
python medical_report_generator_and_translator.py --input_file path/to/your/report.txt --languages "Spanish,Portuguese,Haitian Creole,Vietnamese" --api_key your_openai_api_key
```

c) Use a specific model for translation:

```bash
python medical_report_generator_and_translator.py --model gpt-4o-mini --api_key your_openai_api_key
```

d) Provide guidance for synthetic report generation:

```bash
python medical_report_generator_and_translator.py --guidance "Include information about diabetes" --api_key your_openai_api_key
```

```bash
python medical_report_generator_and_translator.py --guidance "tumour in the left lung and complications due to cirrhosis" --languages russian --api_key your_openai_api_key
```

---

> **Note:** If you have set the `OPENAI_API_KEY` in your environment variables or in a `.env` file in the same directory, you can omit the `--api_key` option.

For more information on available options, run:

```bash
python medical_report_generator_and_translator.py --help
```
