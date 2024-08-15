Installation and Usage Guide

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


Note: If you have set the OPENAI_API_KEY in your environment variables or in a .env file in the same directory, you can omit the --api_key option.

For more information on available options, run:

`python medical_report_generator_and_translator.py --help` 

