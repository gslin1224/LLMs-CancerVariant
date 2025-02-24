import requests
import json
import pandas as pd
import time

# Start timing
start_time = time.time()

# Read the CSV file
file_path = '/path/oncokb_biomarker_drug_associations_1120.csv'
data = pd.read_csv(file_path)

# Select the required columns to construct queries
selected_columns = data[['Gene', 'Alterations', 'Cancer Types']]
if 'Level' in data.columns:
    original_levels = data['Level'].tolist()
else:
    original_levels = ["N/A"] * len(data)  # If the original file does not have a "Level" column, fill in "N/A"

# Construct the query list
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

# Updated system message
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
For example, if the most appropriate level is 1, the next suitable level is 2, and the third is VUS, please answer 1, 2, VUS.
Do not include any additional text or explanations.

### Classification Levels of Evidence:
1: FDA-recognized biomarker predictive of response to an FDA-approved drug in this indication.
2: Standard care biomarker recommended by NCCN or other professional guidelines predictive of response to an FDA-approved drug in this indication.
3A: Compelling clinical evidence supports the biomarker as predictive of response to a drug in this indication.
3B: Standard care or investigational biomarker predictive of response to an FDA-approved or investigational drug in another indication.
4: Compelling biological evidence supports the biomarker as predictive of response to a drug.
R1: Standard care biomarker predictive of resistance to an FDA-approved drug in this indication.
R2: Compelling clinical evidence supports the biomarker as predictive of resistance to a drug.
VUS: Variant of unknown clinical significance; no convincing published evidence of cancer association.
"""

# Batch size
batch_size = 16
epochs = 10  # Set the number of epochs to run

# List of models to test
models = ["qwen2.5:72b"]

# Iterate through each model
for model in models:
    print(f"Testing model: {model}")

    # Create a DataFrame to store all results
    results_df = pd.DataFrame({
        'Query': queries,
        'Original_Level': original_levels
    })
    
    # Run multiple epochs
    for epoch in range(1, epochs + 1):
        responses = []

        # Process queries in batches
        for batch_start in range(0, len(queries), batch_size):
            batch_queries = queries[batch_start:batch_start + batch_size]  # Process batch_size queries at a time
            batch_responses = []
            for i, query in enumerate(batch_queries):
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": ocr_text},
                        {"role": "user", "content": query}
                    ]
                }

                # If the model is GPT-4, add the "family" parameter
                if model == "gpt-4":
                    payload["family"] = "Azure OpenAI"

                # Print the current request being processed
                print(f"Model {model} - Epoch {epoch} - Sending request {batch_start + i + 1}/{len(queries)}: {query}")

                # Send a POST request to the API
                response = requests.post(url, headers=headers, data=json.dumps(payload))
                
                if response.status_code == 200:
                    try:
                        response_data = json.loads(response.text)
                        levels = response_data.get("message", {}).get("content", "").strip()
                        # Print the model's returned value to the console
                        print(f"Model {model} - Epoch {epoch} - Response: {levels}")
                        batch_responses.append(levels)
                    except json.JSONDecodeError:
                        batch_responses.append("Response decoding error")
                        print(f"Model {model} - Epoch {epoch} - Response decoding error for request {batch_start + i + 1}")
                else:
                    error_msg = f"Error: {response.status_code}"
                    batch_responses.append(error_msg)
                    print(f"Model {model} - Epoch {epoch} - Request {batch_start + i + 1} failed, status code: {response.status_code}")
            
            # Append batch responses to the overall response list
            responses.extend(batch_responses)

        # Parse LLM responses and save results into DataFrame
        max_answers = 3  # Assume a maximum of three answers
        for idx, response in enumerate(responses):
            levels = [level.strip() for level in response.split(',') if level.strip()] if isinstance(response, str) else ["N/A"]
            # Handle cases where 1 to 3 answers are returned; fill remaining slots with "N/A"
            for answer_idx in range(max_answers):
                col_name = f"Epoch{epoch}_Level{answer_idx + 1}"
                results_df.loc[idx, f"{model.replace(':', '_')}_{col_name}"] = levels[answer_idx] if answer_idx < len(levels) else "N/A"

    # Save results as a separate Excel file
    output_file_path = f'/path/gene_details_onco_and_civic/onco_details_LLM_{model.replace(":", "_")}_all_epochs_response.xlsx'
    results_df.to_excel(output_file_path, index=False)

    print(f"Results for model {model} saved to: {output_file_path}")

# Calculate and display total execution time
end_time = time.time()
total_time = end_time - start_time
print(f"Total execution time: {total_time:.2f} seconds")
