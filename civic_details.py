import requests
import json
import pandas as pd
import time

# Start timing
start_time = time.time()


file_path = '/path/civic_clear.csv'
data = pd.read_csv(file_path)

# Select the required columns to construct queries
selected_columns = data[['molecular_profile', 'disease']]
if 'evidence_level' in data.columns:  # Ensure the evidence_level column exists
    original_levels = data['evidence_level'].tolist()
else:
    original_levels = ["N/A"] * len(data)  # If the column does not exist, fill in N/A

# Construct the query list
queries = []
for index, row in selected_columns.iterrows():
    query = f"Given the genetic alteration {row['molecular_profile']} in the context of {row['disease']}, what is the associated Level of Evidence?"
    queries.append(query)

# Define API details
url = ""
headers = {
    "Content-Type": "application/json",
    "X-chat-ollama-keys": json.dumps({
        "ollama": {
            "endpoint": "",
            "username": "",
            "password": ""
        },
        "openai": {
            "key": "",
            "endpoint": "",
            "proxy": False
        },
        "azureOpenai": {
            "key": "",
            "endpoint": "",
            "deploymentName": "gpt4o",
            "proxy": False
        },
    })
}

# System message
ocr_text = """

### Role & Objective:
You are an expert assistant specializing in the classification of cancer genetic variants according to clinical significance. Your task is to classify a given gene variant based on its clinical significance using the predefined evidence levels.


### Scope & Behavior
Only classify gene variants based on the provided classification system.
Do not generate explanations, justifications, or interpretations.
Do not provide additional context or assumptions beyond the given query.
If a single classification level is clearly the most suitable, provide only that level.
If the most suitable classification level is not easy to determine, provide up to 3 answers.
List the most suitable classification first, followed by less suitable classifications.


### Input & Output Format
Input Format:
You will receive a natural language query specifying the following elements:
Gene Name (e.g., EGFR, KRAS, BRAF)
Alteration (e.g., L858R, G12D, V600E, amplification, fusion)
Tumor Type (e.g., non-small cell lung cancer, colorectal cancer, melanoma)

Example input:
"Given the gene EGFR, with alteration L858R in the context of non-small cell lung cancer, what is the appropriate classification?"


### Output Format:
If one classification is clearly the best match, return only that classification.
If the classification is uncertain or multiple levels are similarly applicable, provide up to 3 answers.
List the most suitable classification first, followed by less suitable classifications.
Each classification should be separated by commas (,).
For example, if the most appropriate level is A, the next suitable level is B, and the third is VUS, please answer A, B, VUS.
Do not include any additional text or explanations.


### Classification Levels of Evidence:
A: Validated association. Proven/consensus association in human medicine.
Examples:
"AML with mutated NPM1" is a provisional entity in WHO classification of acute myeloid leukemia (AML). This mutation should be tested for in clinical trials and is recommended for testing in patients with cytogenetically normal AML. Validated associations are often in routine clinical practice already or are the subject of major clinical trial efforts.

B: Clinical evidence. Clinical trial or other primary patient data supports association.
Examples:
BRAF V600E is correlated with poor prognosis in papillary thyroid cancer in a study of 187 patients with PTC and other thyroid diseases. The evidence should be supported by observations in multiple patients. Additional support from functional data is desirable but not required.

C: Case study. Individual case reports from clinical journals.
Examples:
A single patient with FLT3 over-expression responded to the FLT3 inhibitor sunitinib. The study may have involved a large number of patients, but the statement was supported by only a single patient. In some cases, observations from just a handful of patients (e.g. 2-3) or a single family may also be considered a case study/report.

D: Preclinical evidence. In vivo or in vitro models support association.
Examples:
Experiments showed that AG1296 is effective in triggering apoptosis in cells with the FLT3 internal tandem duplication. The study may have involved some patient data, but support for this statement was limited to in vivo or in vitro models (e.g. mouse studies, cell lines, molecular assays, etc.).

E: Inferential association. Indirect evidence.
Examples:
CD33 and CD123 expression were significantly increased in patients with NPM1 mutation with FLT3-ITD, indicating these patients may respond to combined anti-CD33 and anti-CD123 therapy. The assertion is at least one step removed from a direct association between a molecular profile (variant) and clinical relevance.

VUS: Variant of unknown clinical significance, No convincing published evidence of cancer association
"""

# Batch size
batch_size = 16
epochs = 10 # Epoch size

# List of models to be tested
models = ["qwen2.5:72b"]

# Iterate through each model
for model in models:
    print(f"Testing model: {model}")

    # Create a DataFrame containing all results
    results_df = pd.DataFrame({
        'Query': queries,
        'Original_Level': original_levels  # Original_Level Column
    })
    
    # Perform multiple epochs
    for epoch in range(1, epochs + 1):
        responses = []

        # Batch processing of queries
        for batch_start in range(0, len(queries), batch_size):
            batch_queries = queries[batch_start:batch_start + batch_size]  # Process batch_size queries
            batch_responses = []
            for i, query in enumerate(batch_queries):
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": ocr_text},
                        {"role": "user", "content": query}
                    ]
                }

                # If the model is gpt-4, use family
                if model == "gpt-4":
                    payload["family"] = "Azure OpenAI"

                # Print the current request being processed
                print(f"Model {model} - Epoch {epoch} - Sending request {batch_start + i + 1}/{len(queries)}: {query}")

                # Send POST request to the API
                response = requests.post(url, headers=headers, data=json.dumps(payload))
                
                if response.status_code == 200:
                    try:
                        response_data = json.loads(response.text)
                        levels = response_data.get("message", {}).get("content", "").strip()
                        # Print the model's return value to the console
                        print(f"Model {model} - Epoch {epoch} - Response: {levels}")
                        batch_responses.append(levels)
                    except json.JSONDecodeError:
                        batch_responses.append("Response decoding error")
                        print(f"Model {model} - Epoch {epoch} - Response decoding error for request {batch_start + i + 1}")
                else:
                    error_msg = f"Error: {response.status_code}"
                    batch_responses.append(error_msg)
                    print(f"Model {model} - Epoch {epoch} - Request {batch_start + i + 1} failed with status code: {response.status_code}")
            
            # Batch responses to response
            responses.extend(batch_responses)

        # Parse LLM responses and store the results in a DataFrame
        max_answers = 3  # Assume a maximum of three answers
        for idx, response in enumerate(responses):
            levels = [level.strip() for level in response.split(',') if level.strip()] if isinstance(response, str) else ["N/A"]
            # Handle cases where 1 to 3 answers are returned, fill with "N/A" if not enough
            for answer_idx in range(max_answers):
                col_name = f"Epoch{epoch}_Level{answer_idx + 1}"
                results_df.loc[idx, f"{model.replace(':', '_')}_{col_name}"] = levels[answer_idx] if answer_idx < len(levels) else "N/A"

    # Save the results as a separate Excel file
    output_file_path = f'/path/gene_details_onco_and_civic/civic_details_LLM_{model.replace(":", "_")}.xlsx'
    results_df.to_excel(output_file_path, index=False)

    print(f"Results for model {model} saved to: {output_file_path}")

# Total execution time
end_time = time.time()
total_time = end_time - start_time
print(f"Total execution time: {total_time:.2f} seconds")
