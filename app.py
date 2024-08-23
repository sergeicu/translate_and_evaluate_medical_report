import streamlit as st
import os
import subprocess
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def run_backend(input_file, languages, cultural_assessment, model, api_key):
    command = [
        "python", "medical_report_generator_and_translator.py",
        "--model", model,
    ]
    
    if input_file:
        command.extend(["--input_file", input_file])
    
    if languages:
        command.extend(["--languages", ",".join(languages)])
    
    if api_key:
        command.extend(["--api_key", api_key])
    
    # Set environment variable for cultural assessment
    os.environ["CULTURAL_ASSESSMENT"] = "1" if cultural_assessment else "0"
    
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout, result.stderr

def main():
    st.title("Medical Report Generator and Translator")

    # File upload
    uploaded_file = st.file_uploader("Drop a file here. If no file provided - synthetic report will be generated", type=["txt"])

    # Language selection
    common_languages = ["Spanish", "Portuguese", "Russian", "Haitian Creole", "Vietnamese", "Chinese (Mandarin)", "Arabic", "French", "Italian", "Greek"]
    selected_languages = []
    st.write("Select languages for translation:")
    for lang in common_languages:
        if st.checkbox(lang):
            selected_languages.append(lang)
    
    # Additional language input
    other_lang = st.text_input("Add other languages (comma-separated)")
    if other_lang:
        selected_languages.extend([lang.strip() for lang in other_lang.split(",")])

    # Cultural assessment checkbox
    cultural_assessment = st.checkbox("Generate cultural assessment")

    # Model selection (for future use)
    model = st.selectbox("Select model", ["gpt-4o-mini"])

    # API key input
    api_key = st.text_input("Enter OpenAI API Key (optional)", type="password")

    if st.button("Generate Report"):
        if not api_key and not os.getenv("OPENAI_API_KEY"):
            st.error("Please provide an API key or set it in the .env file")
        else:
            with st.spinner("Generating report...this may take a minute"):
                input_file = uploaded_file.name if uploaded_file else None
                stdout, stderr = run_backend(input_file, selected_languages, cultural_assessment, model, api_key)
                
                st.subheader("Progress:")
                st.text(stdout)
                
                if stderr:
                    st.error("Errors:")
                    st.text(stderr)
                
                # Find the report folder
                report_folder = None
                for line in stdout.split("\n"):
                    if line.startswith("Report generation completed. Final report folder:"):
                        report_folder = line.split(":")[1].strip()
                        break
                
                if report_folder:
                    st.subheader("Generated Files:")
                    for root, dirs, files in os.walk(report_folder):
                        for file in files:
                            file_path = os.path.join(root, file)
                            relative_path = os.path.relpath(file_path, report_folder)
                            if file.endswith(('.txt', '.json')):
                                if st.button(f"View {relative_path}"):
                                    with open(file_path, 'r') as f:
                                        content = f.read()
                                        if file.endswith('.json'):
                                            st.json(json.loads(content))
                                        else:
                                            st.text(content)

if __name__ == "__main__":
    main()