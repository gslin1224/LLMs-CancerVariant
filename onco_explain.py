import requests
import json
import pandas as pd
import time
import os

# Start timing
start_time = time.time()

# Read the dataset
file_path = '/path/gene_explain/oncokb_biomarker_drug_associations_1120_SelectedForLLMExplanation.xlsx'
data = pd.read_excel(file_path)

# Select the required columns to construct queries
selected_columns = data[['Gene', 'Alterations', 'Cancer Types']]
if 'Level' in data.columns:
    original_levels = data['Level'].tolist()
else:
    original_levels = ["N/A"] * len(data)  # If the original file does not contain the "Level" column, fill in "N/A"

# Construct a list of queries
queries = []
for index, row in selected_columns.iterrows():
    query = f"Given the gene {row['Gene']}, with alteration {row['Alterations']} in the context of {row['Cancer Types']}, what is the associated Level?"
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

# Prompt is left blank for user input
ocr_text = """
### Instructions:
Answer with the level number corresponding to the classification and explain the reasoning behind your answer.

You can provide one or more appropriate levels as the answer, up to a maximum of three, and please arrange them in order of suitability.

Level of Evidence:

1: FDA-recognized biomarker predictive of response to an FDA-approved drug in this indication

2: Standard care biomarker recommended by the NCCN or other professional guidelines predictive of response to an FDA-approved drug in this indication

3A: Compelling clinical evidence supports the biomarker as being predictive of response to a drug in this indication

3B: Standard care or investigational biomarker predictive of response to an FDA-approved or investigational drug in another indication

4: Compelling biological evidence supports the biomarker as being predictive of response to a drug

R1: Standard care biomarker predictive of resistance to an FDA-approved drug in this indication

R2: Compelling clinical evidence supports the biomarker as being predictive of resistance to a drug

VUS: Variant of unknown clinical significance, No convincing published evidence of cancer association

"""  # The user can fill this in

# Multi-model processing
models = ["gpt-4", "qwen2.5:72b", "llama3.1:70b"]
log_dir = "/path/gene_explain/OncoKB"
os.makedirs(log_dir, exist_ok=True)

# Create a DataFrame to store all results
results_df = pd.DataFrame({
    'Query': queries,
    'Original_Level': original_levels
})

# Batch size
batch_size = 5

# Iterate through each model
for model in models:
    print(f"Testing model: {model}")

    log_file_path = os.path.join(log_dir, f"oncokb_LLM_log_{model}.log")

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

    # Store responses in the DataFrame
    results_df[model.replace(':', '_')] = responses

# Save combined results
output_file_path = os.path.join(log_dir, "oncokb_LLM_results_combined.xlsx")
results_df.to_excel(output_file_path, index=False)
print(f"Results from all models saved to: {output_file_path}")

# Calculate total execution time
end_time = time.time()
total_time = end_time - start_time
print(f"Total execution time: {total_time:.2f} seconds")
