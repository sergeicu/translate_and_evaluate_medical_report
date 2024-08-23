import streamlit as st
import os
import json
from dotenv import load_dotenv
import base64
from medical_report_generator_and_translator import run_report_generation

# Load environment variables
load_dotenv()

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{file_label}</a>'
    return href

def main():
    st.title("Medical Report Generator and Translator")

    if 'report_generated' not in st.session_state:
        st.session_state.report_generated = False
    if 'file_viewers' not in st.session_state:
        st.session_state.file_viewers = {}
    if 'debug' not in st.session_state:
        st.session_state.debug = False
    if 'input_data' not in st.session_state:
        st.session_state.input_data = {}

    # File upload
    uploaded_file = st.file_uploader("Drop a file here. If no file provided - synthetic report will be generated", type=["txt"])

    # Language selection
    common_languages = ["Spanish", "Portuguese", "Russian", "Haitian Creole", "Vietnamese", "Chinese (Mandarin)", "Arabic", "French", "Italian", "Greek"]
    selected_languages = []
    st.write("Select languages for translation:")
    for lang in common_languages:
        if st.checkbox(lang, key=f"lang_{lang}"):
            selected_languages.append(lang)
    
    # Additional language input
    other_lang = st.text_input("Add other languages (comma-separated)")
    if other_lang:
        selected_languages.extend([lang.strip() for lang in other_lang.split(",")])

    # Cultural assessment checkbox
    cultural_assessment = st.checkbox("Generate cultural assessment")

    # Developer Mode section
    with st.expander("Developer Mode"):
        # Model selection
        model = st.selectbox("Select model", ["gpt-4o-mini"])

        # API key input
        api_key = st.text_input("Enter OpenAI API Key (optional)", type="password")

        # Debug checkbox
        debug = st.checkbox("Debug")

        # Micro report checkbox
        micro_report = st.checkbox("Micro report")

    if not st.session_state.report_generated:
        if st.button("Generate Report"):
            if not api_key and not os.getenv("OPENAI_API_KEY"):
                st.error("Please provide an API key or set it in the .env file")
            else:
                with st.spinner("Generating report..."):
                    try:
                        input_content = uploaded_file.getvalue().decode('utf-8') if uploaded_file else None
                        os.environ["CULTURAL_ASSESSMENT"] = "1" if cultural_assessment else "0"
                        
                        report_folder = run_report_generation(
                            input_content=input_content,
                            languages=",".join(selected_languages),
                            model=model,
                            api_key=api_key,
                            microreport=micro_report
                        )
                        
                        st.session_state.report_generated = True
                        st.session_state.report_folder = report_folder
                        st.session_state.input_data = {
                            "uploaded_file": uploaded_file,
                            "selected_languages": selected_languages,
                            "cultural_assessment": cultural_assessment,
                            "model": model,
                            "api_key": api_key,
                            "debug": debug,
                            "micro_report": micro_report
                        }
                        st.rerun()
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

    if st.session_state.report_generated:
        st.markdown("---")

        if st.session_state.debug:
            st.subheader("Debug Information:")
            st.json(st.session_state.input_data)
        
        report_folder = st.session_state.report_folder
        
        if report_folder:
            st.subheader("Generated Files:")
            all_files = []
            for root, dirs, files in os.walk(report_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, report_folder)
                    if file.endswith(('.txt', '.json')):
                        all_files.append(file_path)
                        
                        # Display files based on debug mode
                        if st.session_state.debug or (not file.startswith("params/")):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                if st.button(f"{relative_path}"):
                                    if relative_path not in st.session_state.file_viewers:
                                        st.session_state.file_viewers[relative_path] = True
                                    else:
                                        st.session_state.file_viewers[relative_path] = not st.session_state.file_viewers[relative_path]
                            with col2:
                                st.markdown(get_binary_file_downloader_html(file_path, "Download"), unsafe_allow_html=True)
                            
                            if relative_path in st.session_state.file_viewers and st.session_state.file_viewers[relative_path]:
                                with open(file_path, 'r') as f:
                                    content = f.read()
                                    if file.endswith('.json'):
                                        st.json(json.loads(content))
                                    else:
                                        st.text_area("File content", content, height=300)
        
        st.markdown("---")
        if st.button("Generate New Report", key="new_report", type="primary"):
            st.session_state.report_generated = False
            st.session_state.file_viewers = {}
            st.rerun()

if __name__ == "__main__":
    main()