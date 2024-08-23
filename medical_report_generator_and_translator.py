"""Medical Report Generator and Translator

This script generates synthetic medical reports and translates them into multiple languages,
then evaluates the quality of the translations using various metrics. It provides a comprehensive
tool for creating, translating, and assessing medical reports across different languages.

Usage:
    python medical_report_generator_and_translator.py [options]

Options:
    --input_file TEXT         Path to a .txt file for translation (optional)
    --languages TEXT          Comma-separated list of target languages for translation
    --model TEXT              Model to use for translation (default: gpt-4o-mini)
    --guidance TEXT           Specifications for synthetic report generation (optional)
    --api_key TEXT            OpenAI API key (required if not in .env file)

Note: Ensure that you have set the OPENAI_API_KEY environment variable before running the script.



The script uses the following evaluation metrics:

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



"""




import os
import json
import glob
import re
import argparse
import csv
from typing import List, Dict, Any
from datetime import datetime
import shutil

import torch
import numpy as np
from litellm import completion
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.ribes_score import sentence_ribes
from textstat import flesch_reading_ease, smog_index
from dotenv import load_dotenv

import nltk 
nltk.download('wordnet')

# Initialize BERT model for semantic similarity
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',  clean_up_tokenization_spaces=True)  # or False
model = BertModel.from_pretrained('bert-base-uncased')

# Define MEDICAL_TERMS (placeholder, should be replaced with actual terms)
MEDICAL_TERMS = ["anemia", "hypertension", "diabetes", "arrhythmia", "hypothyroidism"]

def generate_medical_report(report_folder: str, guidance: str = None) -> str:
    """
    Generate a synthetic medical report for a fictional patient.

    Args:
        report_folder (str): The folder to save the report and parameters.
        guidance (str, optional): Specific instructions or parameters for report generation.

    Returns:
        str: The generated medical report content.
    """
    prompt = """Generate a random medical report for a fictional patient. Include the following sections:
    1. Patient Information
    2. Chief Complaint
    3. History of Present Illness
    4. Past Medical History
    5. Physical Examination
    6. Assessment
    7. Plan

    Please make sure the information is realistic but entirely fictional."""

    if guidance:
        prompt += f"\n\nPlease incorporate the following guidance in the report generation: {guidance}"

    try:
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        report_content = response.choices[0].message.content
        save_report(report_content, os.path.join(report_folder, "original_report.txt"))
        # save_report(report_content, os.path.join(report_folder, "original_report.json"), format="json")
        # save_report(report_content, os.path.join(report_folder, "original_report.csv"), format="csv")

        params = {
            "prompt": prompt,
            "model": "gpt-3.5-turbo",
        }
        params_folder = os.path.join(report_folder, "params")
        os.makedirs(params_folder, exist_ok=True)
        save_params(params, os.path.join(params_folder, "generation_params.json"))

        return report_content
    except Exception as e:
        print(f"Error generating medical report: {str(e)}")
        return None

def translate_report(report: str, target_language: str, report_folder: str, model: str = "gpt-4o-mini") -> str:
    """
    Translate a medical report into the specified target language.

    Args:
        report (str): The original medical report content to be translated.
        target_language (str): The language to translate the report into.
        report_folder (str): The folder to save the translated report and parameters.
        model (str): The model to use for translation (default: "gpt-4o-mini").

    Returns:
        str: The translated medical report content.
    """
    prompt = f"Translate the following medical report into {target_language}. Maintain the original formatting and structure. Ensure that medical terms are accurately translated and cultural sensitivities are considered:\n\n{report}"

    try:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        translated_content = response.choices[0].message.content
        save_report(translated_content, os.path.join(report_folder, f"translated_report_{target_language}.txt"))
        # save_report(translated_content, os.path.join(report_folder, f"translated_report_{target_language}.json"), format="json")
        # save_report(translated_content, os.path.join(report_folder, f"translated_report_{target_language}.csv"), format="csv")

        params = {
            "prompt": prompt,
            "model": model,
        }
        params_folder = os.path.join(report_folder, "params")
        os.makedirs(params_folder, exist_ok=True)
        save_params(params, os.path.join(params_folder, f"translation_params_{target_language}.json"))

        return translated_content
    except Exception as e:
        print(f"Error translating report: {str(e)}")
        return None

def evaluate_translation(original: str, translated: str, target_language: str, report_folder: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Evaluate the quality of a translated medical report using various metrics.

    Args:
        original (str): The original medical report content in English.
        translated (str): The translated medical report content.
        target_language (str): The language of the translated report.
        report_folder (str): The folder to save the evaluation metrics.
        model (str): The model to use for back-translation (default: "gpt-4o-mini").

    Returns:
        Dict[str, Any]: A dictionary containing all the calculated metrics.
    """
    try:
        # Get back-translation
        back_translation_prompt = f"Translate the following {target_language} medical report back to English, maintaining the original structure and formatting:\n\n{translated}"
        back_translation_response = completion(
            model=model,
            messages=[{"role": "user", "content": back_translation_prompt}],
        )
        back_translation = back_translation_response.choices[0].message.content

        # Calculate metrics
        bleu = calculate_bleu(original, back_translation)
        meteor = calculate_meteor(original, back_translation)
        ribes = calculate_ribes(original, back_translation)
        bert_score = calculate_bert_score(original, back_translation)
        readability_original = calculate_readability(original)
        readability_translated = calculate_readability(translated)
        terminology_accuracy = check_terminology_accuracy(original, translated, back_translation)
        consistency = check_consistency(translated)
        errors = categorize_errors(original, back_translation)

        # Prepare metrics dictionary
        metrics = {
            "BLEU Score": bleu,
            "METEOR Score": meteor,
            "RIBES Score": ribes,
            "BERTScore": bert_score,
            "Readability (Original)": readability_original,
            "Readability (Translated)": readability_translated,
            "Terminology Accuracy": terminology_accuracy,
            "Consistency": consistency,
            "Errors": errors
        }

        # Save metrics
        # save_report(metrics, os.path.join(report_folder, f"metrics_{target_language}.txt"), format="txt")
        save_report(metrics, os.path.join(report_folder, f"metrics_{target_language}.json"), format="json")
        # save_report(metrics, os.path.join(report_folder, f"metrics_{target_language}.csv"), format="csv")

        return metrics
    except Exception as e:
        print(f"Error evaluating translation: {str(e)}")
        return None

def assess_cultural_appropriateness(translated: str, target_language: str, report_folder: str, model: str = "gpt-4o-mini") -> str:
    """
    Assess the cultural appropriateness of a translated medical report.

    Args:
        translated (str): The translated medical report content to be assessed.
        target_language (str): The language of the translated report.
        report_folder (str): The folder to save the assessment.
        model (str): The model to use for assessment (default: "gpt-4o-mini").

    Returns:
        str: A brief assessment of the cultural appropriateness and recommendations for improvement.
    """
    prompt = f"""Assess the cultural appropriateness of the following medical report translation in {target_language}. 
    Consider factors such as:
    1. Use of appropriate honorifics or forms of address
    2. Sensitivity to cultural taboos or stigmas related to health
    3. Appropriate use of idiomatic expressions
    4. Consideration of cultural beliefs about health and medicine

    Report:
    {translated}

    Provide a brief assessment and any recommendations for improvement."""

    try:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        assessment = response.choices[0].message.content
        save_report(assessment, os.path.join(report_folder, f"cultural_assessment_{target_language}.txt"))
        # save_report(assessment, os.path.join(report_folder, f"cultural_assessment_{target_language}.json"), format="json")
        # save_report(assessment, os.path.join(report_folder, f"cultural_assessment_{target_language}.csv"), format="csv")
        return assessment
    except Exception as e:
        print(f"Error assessing cultural appropriateness: {str(e)}")
        return None

def get_report_folder(input_file: str = None) -> str:
    """
    Get the folder name for saving the report and its translations.

    Args:
        input_file (str, optional): The input file name, if provided.

    Returns:
        str: The folder name for the report.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if input_file:
        basename = os.path.splitext(os.path.basename(input_file))[0]
        folder_name = f"report_{basename}_{timestamp}_in_progress"
    else:
        existing_reports = glob.glob("report_*/")
        report_numbers = [int(report.split("_")[1]) for report in existing_reports if report.split("_")[1].isdigit()]
        next_number = max(report_numbers) + 1 if report_numbers else 1
        folder_name = f"report_{next_number:03d}_{timestamp}_in_progress"
    
    os.makedirs(folder_name, exist_ok=True)
    # os.makedirs(os.path.join(folder_name, "csv"), exist_ok=True)
    os.makedirs(os.path.join(folder_name, "json"), exist_ok=True)
    os.makedirs(os.path.join(folder_name, "txt"), exist_ok=True)
    return folder_name

def save_report(content: Any, filename: str, format: str = "txt") -> None:
    """
    Save the report content to a file in the specified format.

    Args:
        content (Any): The content to be saved.
        filename (str): The name of the file to save the content to.
        format (str): The format to save the content in ("txt", "json", or "csv").
    """
    try:
        folder, basename = os.path.split(filename)
        subfolder = os.path.join(folder, format)
        os.makedirs(subfolder, exist_ok=True)
        full_path = os.path.join(subfolder, basename)

        if format == "txt":
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(str(content))
        elif format == "json":
            # Convert any numpy.float32 to regular float
            def convert_floats(obj):
                if isinstance(obj, np.float32):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_floats(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_floats(i) for i in obj]
                return obj

            content = convert_floats(content)
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
        elif format == "csv":
            with open(full_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if isinstance(content, dict):
                    writer.writerow(content.keys())
                    writer.writerow(content.values())
                elif isinstance(content, str):
                    writer.writerow([content])
                else:
                    writer.writerow([str(content)])
    except Exception as e:
        print(f"Error saving report: {str(e)}")

def save_params(params: Dict[str, Any], filename: str) -> None:
    """
    Save parameters to a JSON file.

    Args:
        params (dict): The parameters to be saved.
        filename (str): The name of the file to save the parameters to.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2)
    except Exception as e:
        print(f"Error saving parameters: {str(e)}")

def calculate_bleu(reference: str, hypothesis: str) -> float:
    """
    Calculate the BLEU score between a reference and hypothesis translation.

    Args:
        reference (str): The reference (original) text.
        hypothesis (str): The hypothesis (translated) text.

    Returns:
        float: The BLEU score.
    """
    return sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=SmoothingFunction().method1)

def calculate_meteor(reference: str, hypothesis: str) -> float:
    """
    Calculate the METEOR score between a reference and hypothesis translation.

    Args:
        reference (str): The reference (original) text.
        hypothesis (str): The hypothesis (translated) text.

    Returns:
        float: The METEOR score.
    """
    return meteor_score.meteor_score([reference.split()], hypothesis.split())

def calculate_ribes(reference: str, hypothesis: str) -> float:
    """
    Calculate the RIBES score between a reference and hypothesis translation.

    Args:
        reference (str): The reference (original) text.
        hypothesis (str): The hypothesis (translated) text.

    Returns:
        float: The RIBES score.
    """
    return sentence_ribes(reference.split(), hypothesis.split())

def calculate_bert_score(reference: str, hypothesis: str) -> float:
    """
    Calculate the BERTScore between a reference and hypothesis translation.

    Args:
        reference (str): The reference (original) text.
        hypothesis (str): The hypothesis (translated) text.

    Returns:
        float: The BERTScore (cosine similarity of BERT embeddings).
    """
    ref_encoding = tokenizer(reference, return_tensors='pt', padding=True, truncation=True)
    hyp_encoding = tokenizer(hypothesis, return_tensors='pt', padding=True, truncation=True)

    with torch.no_grad():
        ref_outputs = model(**ref_encoding)
        hyp_outputs = model(**hyp_encoding)

    ref_embeddings = ref_outputs.last_hidden_state.mean(dim=1)
    hyp_embeddings = hyp_outputs.last_hidden_state.mean(dim=1)

    similarity = cosine_similarity(ref_embeddings, hyp_embeddings)
    return similarity[0][0]

def calculate_readability(text: str) -> Dict[str, float]:
    """
    Calculate the readability scores of a given text.

    Args:
        text (str): The text to evaluate for readability.

    Returns:
        dict: A dictionary containing the Flesch Reading Ease and SMOG Index scores.
    """
    return {
        "flesch": flesch_reading_ease(text),
        "smog": smog_index(text)
    }

def check_terminology_accuracy(original: str, translated: str, back_translated: str) -> float:
    """
    Check the accuracy of medical terminology in the translation.

    Args:
        original (str): The original text.
        translated (str): The translated text.
        back_translated (str): The back-translated text.

    Returns:
        float: The terminology accuracy score (0-1).
    """
    original_terms = [term.lower() for term in MEDICAL_TERMS if term.lower() in original.lower()]
    back_translated_terms = [term.lower() for term in MEDICAL_TERMS if term.lower() in back_translated.lower()]
    accuracy = len(set(original_terms) & set(back_translated_terms)) / len(original_terms) if original_terms else 1
    return accuracy

def check_consistency(translated: str) -> Dict[str, int]:
    """
    Check the consistency of medical term translations.

    Args:
        translated (str): The translated text to check for consistency.

    Returns:
        dict: A dictionary with medical terms as keys and their counts as values.
    """
    term_counts = {}
    for term in MEDICAL_TERMS:
        term_counts[term] = translated.lower().count(term.lower())
    return term_counts

def categorize_errors(original: str, back_translated: str) -> Dict[str, List[str]]:
    """
    Categorize potential errors in the translation by comparing the original and back-translated texts.

    Args:
        original (str): The original text.
        back_translated (str): The back-translated text.

    Returns:
        dict: A dictionary with error categories as keys and lists of identified errors as values.
    """
    errors = {
        "terminology": [],
        "grammar": [],
        "style": [],
        "omission": [],
        "addition": []
    }
    
    original_sentences = original.split('.')
    back_translated_sentences = back_translated.split('.')

    for orig, back in zip(original_sentences, back_translated_sentences):
        if not set(orig.split()) == set(back.split()):
            for term in MEDICAL_TERMS:
                if term.lower() in orig.lower() and term.lower() not in back.lower():
                    errors["terminology"].append(f"Missing term: {term}")
            if len(orig.split()) > len(back.split()):
                errors["omission"].append(f"Possible omission in: {back}")
            elif len(orig.split()) < len(back.split()):
                errors["addition"].append(f"Possible addition in: {back}")
            if orig.count(',') != back.count(','):
                errors["grammar"].append(f"Comma mismatch in: {back}")
            if orig.isupper() != back.isupper():
                errors["style"].append(f"Capitalization mismatch in: {back}")

    return errors
def save_translations_csv(report_folder: str, original_report: str, translations: Dict[str, str]) -> None:
    """
    Save all generated translations and the original report to a CSV file.

    Args:
        report_folder (str): The folder where the report files are saved.
        original_report (str): The original report content.
        translations (Dict[str, str]): A dictionary of translations with language codes as keys.
    """
    for format in ['json']: #['csv', 'json', 'txt']:
        filename = os.path.join(report_folder, f"all_translations.{format}")
        content = {
            'English (Original)': original_report,
            **translations
        }
        save_report(content, filename, format)

def save_metrics_csv(report_folder: str, metrics: Dict[str, Dict[str, Any]]) -> None:
    """
    Save all metrics for each translation to a CSV file.

    Args:
        report_folder (str): The folder where the report files are saved.
        metrics (Dict[str, Dict[str, Any]]): A dictionary of metrics for each translation.
    """
    for format in ['json']: # ['csv', 'json', 'txt']:
        filename = os.path.join(report_folder, f"all_metrics.{format}")
        save_report(metrics, filename, format)

def check_json_completeness(json_file: str) -> bool:
    """
    Check if the generated JSON file is complete and valid.

    Args:
        json_file (str): Path to the JSON file to check.

    Returns:
        bool: True if the JSON file is complete and valid, False otherwise.
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            json.load(f)
        return True
    except json.JSONDecodeError:
        return False
    except FileNotFoundError:
        print(f"File not found: {json_file}")
        return False


def evaluate_translation_quality(metrics: Dict[str, Dict[str, Any]], model: str = "gpt-4o-mini") -> Dict[str, str]:
    """
    Evaluate the overall quality of translations based on the metrics using GPT-4.

    Args:
        metrics (Dict[str, Dict[str, Any]]): A dictionary of metrics for each translation.
        model (str): The model to use for evaluation (default: "gpt-4o-mini").

    Returns:
        Dict[str, str]: A dictionary containing the overall evaluation and justification.
    """
    overall_evaluation = {}
    for language, metric_values in metrics.items():
        
        # Convert metrics to native Python types
        metric_values_native = convert_to_native_types(metric_values)
                
        prompt = f"""Evaluate the quality of the following translation based on these metrics:
        {json.dumps(metric_values_native, indent=2)}
        
        Rate the quality of the translation on a scale of 1-5, where 1 is very poor and 5 is excellent.
        Provide a short summary statement justifying the evaluation score."""
        
        try:
            response = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            evaluation = response.choices[0].message.content.strip()
            overall_evaluation[language] = evaluation
        except Exception as e:
            print(f"Error evaluating translation quality for {language}: {str(e)}")
            overall_evaluation[language] = "Evaluation failed."
    
    return overall_evaluation

def convert_to_native_types(obj):
    """
    Recursively convert numpy data types to native Python types.
    """
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(i) for i in obj]
    return obj


def main():
    print(f"Analyzing...\n\n\n")
    parser = argparse.ArgumentParser(description="Medical Report Generator and Translator")
    parser.add_argument("--input_file", type=str, help="Path to a .txt file for translation")
    parser.add_argument("--languages", type=str, default="Haitian Creole,Chinese Mandarin,Vietnamese,Russian,Arabic",
                        help="Comma-separated list of target languages for translation")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use for translation")
    parser.add_argument("--guidance", type=str, help="Specifications for synthetic report generation")
    parser.add_argument("--api_key", type=str, help="OpenAI API key")
    
    args = parser.parse_args()

    # Load .env file
    load_dotenv()

    # Set OpenAI API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key must be provided either in .env file or via --api_key argument.")
        return

    os.environ["OPENAI_API_KEY"] = api_key

    report_folder = get_report_folder(args.input_file)

    if args.input_file:
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                original_report = f.read()
        except Exception as e:
            print(f"Error reading input file: {str(e)}")
            return
    else:
        print("IMPORTANT: The system is generating synthetic report. \nIf you want to generate report from file - please provide file path via --input_file argument")
        original_report = generate_medical_report(report_folder, args.guidance)

    if original_report is None:
        print("Failed to generate or read the original report. Exiting.")
        return

    print(f"Original Report (English) saved in {report_folder}")
    print("\n" + "="*50 + "\n")

    languages = args.languages.split(',')
    translations = {}
    all_metrics = {}

    for language in languages:
        translated_report = translate_report(original_report, language, report_folder, args.model)
        if translated_report is None:
            print(f"Failed to translate report to {language}. Skipping evaluation for this language.")
            continue
        
        translations[language] = translated_report
        print(f"Translated Report ({language}) saved in {report_folder}")
        
        metrics = evaluate_translation(original_report, translated_report, language, report_folder, args.model)
        if metrics:
            all_metrics[language] = metrics
            print(f"Translation metrics for {language} saved in {report_folder}")
        else:
            print(f"Failed to evaluate translation for {language}.")
        
        # Check if cultural assessment is enabled
        if os.environ.get("CULTURAL_ASSESSMENT") == "1":
            cultural_assessment = assess_cultural_appropriateness(translated_report, language, report_folder, args.model)
            if cultural_assessment:
                print(f"Cultural appropriateness assessment for {language} saved in {report_folder}")
            else:
                print(f"Failed to assess cultural appropriateness for {language}.")
        
        print("\n" + "="*50 + "\n")

    # Add metrics for the original report (English)
    all_metrics['English'] = {'Readability (Original)': calculate_readability(original_report)}

    # Save translations and metrics to separate files
    save_translations_csv(report_folder, original_report, translations)
    save_metrics_csv(report_folder, all_metrics)
    # print(f"All translations saved in {report_folder}")
    # print(f"All metrics saved in {report_folder}")


    # Evaluate the overall translation quality
    overall_evaluation = evaluate_translation_quality(all_metrics, args.model)

    # Save the evaluation to a .txt file
    evaluation_file = os.path.join(report_folder, "overall_evaluation.txt")
    with open(evaluation_file, 'w', encoding='utf-8') as f:
        for language, evaluation in overall_evaluation.items():
            f.write(f"Language: {language}\n")
            f.write(f"Evaluation: {evaluation}\n\n")

    print(f"Overall translation evaluation saved in {evaluation_file}")        

    # # Check JSON completeness
    # json_file = os.path.join(report_folder, "json", "original_report.json")
    # if check_json_completeness(json_file):
    #     print(f"JSON file {json_file} is complete and valid.")
    # else:
    #     print(f"JSON file {json_file} is incomplete or invalid.")
            


    # Remove "_in_progress" suffix from the folder name
    new_folder_name = report_folder.replace("_in_progress", "")
    os.rename(report_folder, new_folder_name)
    print(f"Report generation completed. Final report folder: {new_folder_name}")
    
        

if __name__ == "__main__":
    main()