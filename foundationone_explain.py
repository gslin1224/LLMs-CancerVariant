import requests
import json
import pandas as pd
import time
import os

# Start timing
start_time = time.time()

# Read the FoundationOne dataset
file_path = '/path/00_FoundationOne_Variants_Summary_for_LLM_Test.xlsx'
data = pd.read_excel(file_path, sheet_name='FoundationOne_Variants_Summary') 
data = data.head(100)

# Select the required columns to construct queries
selected_columns = data[['gene', 'Variant', 'TumorType']]
original_levels = data['Level'].tolist()

# Construct the query list
queries = []
for index, row in selected_columns.iterrows():
    query = f"Given the gene {row['gene']}, with alteration {row['Variant']} in the context of {row['TumorType']}, what is the appropriate classification?"
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

# New system message
ocr_text = """
### Instructions:
Answer with the level number corresponding to the classification and explain the reasoning behind your answer.

You can provide one or more appropriate levels as the answer, up to a maximum of three, and please arrange them in order of suitability.

Level of Evidence:

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

# Multi-model processing
models = ["qwen2.5:72b"]
log_dir = "/path/gene_explain/FoundationOneCivic_Originly"
os.makedirs(log_dir, exist_ok=True)

# Create a DataFrame containing all results
results_df = pd.DataFrame({
    'Query': queries,
    'Original_Level': original_levels
})

# Batch size
batch_size = 16

# Iterate through each model
for model in models:
    print(f"Testing model: {model}")

    log_file_path = os.path.join(log_dir, f"foundation_LLM_log_{model}.log")

    # Initialize log file
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"Model: {model}\n")
        log_file.write(f"Total Questions: {len(queries)}\n")
        log_file.write("-" * 50 + "\n")

    responses = []

    # Process all queries in batches
    for batch_start in range(0, len(queries), batch_size):
        batch_queries = queries[batch_start:batch_start + batch_size]
        batch_responses = []

        for i, query in enumerate(batch_queries):
            question_number = batch_start + i + 1
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": ocr_text},
                    {"role": "user", "content": query}
                ]
            }

            if model == "gpt-4":
                payload["family"] = "Azure OpenAI"

            # Send request and log response
            try:
                response = requests.post(url, headers=headers, data=json.dumps(payload))
                if response.status_code == 200:
                    response_data = json.loads(response.text)
                    response_text = response_data.get("message", {}).get("content", "").strip()
                    batch_responses.append(response_text)
                    with open(log_file_path, 'a') as log_file:
                        log_file.write(f"Question #{question_number}/{len(queries)} Query: {query}\nResponse: {response_text}\n\n")
                else:
                    error_msg = f"Error: {response.status_code}"
                    batch_responses.append(error_msg)
                    with open(log_file_path, 'a') as log_file:
                        log_file.write(f"Question #{question_number}/{len(queries)} Query: {query}\nResponse: {error_msg}\n\n")
            except Exception as e:
                batch_responses.append(f"Request failed: {e}")
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f"Question #{question_number}/{len(queries)} Query: {query}\nResponse: Request failed: {e}\n\n")

        responses.extend(batch_responses)

    # Store responses in DataFrame
    results_df[model.replace(':', '_')] = responses

# Save the combined results
output_file_path = os.path.join(log_dir, "foundation_LLM_results_Explain_origin.xlsx")
results_df.to_excel(output_file_path, index=False)
print(f"Results for all models have been saved to: {output_file_path}")

# Calculate total execution time
end_time = time.time()
total_time = end_time - start_time
print(f"Total execution time: {total_time:.2f} seconds")
