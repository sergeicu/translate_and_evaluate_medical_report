# Medical Report Generator and Translator

A comprehensive tool for creating, translating, and evaluating translation of medical reports across different languages.

In detail:  

Generate synthetic medical reports (or ingest existing .txt files), translate them into multiple languages using any LLM, then evaluate the quality of the translations using various metrics. 

Usage:
    python medical_report_generator_and_translator.py [options]

Output:
    Create a new folder with translation for each language specified in .csv, .txt and .json format, together with corresponding evaluation report. 

Options:
    --languages TEXT          Comma-separated list of target languages for translation
    --model TEXT              LLM model to use for translation (default: gpt-4o-mini)
    --guidance TEXT           Additional specification for synthetic report generation - such as socio-demographics or disease details
    --input_file TEXT         Path to a .txt file with medical report (to use instead of synthetic report)
    --api_key TEXT            OpenAI API key (required if not present in .env file)

Note: Ensure that you have OPENAI_API_KEY in .env file before running the script, or provide it via --api_key argument. 



The script uses the following translation evaluation metrics:

1. BLEU (Bilingual Evaluation Understudy):
   Measures the precision of n-gram matches between the translated text and reference text.
   Important for assessing overall translation quality.

2. METEOR (Metric for Evaluation of Translation with Explicit ORdering):
   Considers synonyms and paraphrases, which is particularly useful for medical terminology.
   Provides a more nuanced evaluation of semantic similarity.

3. RIBES (Rank-based Intuitive Bilingual Evaluation Score):
   Evaluates word order in translations, which is crucial for maintaining the logical flow
   of medical information.

4. BERTScore:
   Uses contextual embeddings to compute similarity, capturing semantic meaning better than
   exact matches. Particularly useful for assessing the preservation of medical concepts.

5. Readability Metrics (Flesch Reading Ease):
   Ensures that the translated text maintains an appropriate level of readability, which is
   crucial for patient understanding in medical contexts.

6. Terminology Accuracy:
   Checks the accuracy of key medical terms in the translation, ensuring that critical
   medical information is preserved.

7. Consistency Check:
   Evaluates how consistently specific terms or phrases are translated throughout the document,
   which is essential for maintaining clarity in medical reports.

8. Error Categorization:
   Categorizes errors (e.g., terminology, grammar, style) to provide more actionable feedback
   for improving translations.

9. Cultural Appropriateness Assessment:
   Evaluates the cultural sensitivity and appropriateness of the translations, which is
   crucial in healthcare communication across different cultures.



---

# Installation and Usage Guide

This guide provides instructions on how to install the required packages and run the Medical Report Generator and Translator.

1. Installation:

Ensure you have Python 3.7+ installed on your system. Then, install the required packages using pip:

`pip install -r requirements.txt`


2. Usage:

To run the script, use the following command format:

`python medical_report_generator_and_translator.py [options]`

Examples:

a) Generate a synthetic report and translate it to default languages:

`python medical_report_generator_and_translator.py --api_key your_openai_api_key` 

b) Translate an existing report to specific languages:

`python medical_report_generator_and_translator.py --input_file path/to/your/report.txt --languages "Spanish,Portguese,Haitian Creole,Vietnamese" --api_key your_openai_api_key`

c) Use a specific model for translation:


`python medical_report_generator_and_translator.py --model gpt-4o-mini --api_key your_openai_api_key`

d) Provide guidance for synthetic report generation:


`python medical_report_generator_and_translator.py --guidance "Include information about diabetes" --api_key your_openai_api_key`

`python medical_report_generator_and_translator.py --guidance "tumour in the left lung and complications due to cirhosis" --languages russian --api_key your_openai_api_key`

--- 

Note: If you have set the OPENAI_API_KEY in your environment variables or in a .env file in the same directory, you can omit the --api_key option.

For more information on available options, run:

`python medical_report_generator_and_translator.py --help` 

