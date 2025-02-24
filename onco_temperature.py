import requests
import json
import pandas as pd
import time

# Start timing
start_time = time.time()

# Read the CSV file
file_path = '/path/gene_onco/oncokb_biomarker_drug_associations_1120.csv'
data = pd.read_csv(file_path)

# Select the required columns to construct queries
selected_columns = data[['Gene', 'Alterations', 'Cancer Types']]
if 'Level' in data.columns:
    original_levels = data['Level'].tolist()
else:
    original_levels = ["N/A"] * len(data)  # If the original file does not have a Level column, fill in N/A

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
### Instructions:
You should select one, two, or three appropriate levels based on the specific context of the given query. Your response must consist only of the levels that are most relevant and applicable to the provided gene, alteration, and cancer type, separated by commas. No additional text, explanations, reasoning, or context should be included.

Examples:
- If the context indicates that only one level is most suitable, respond with that level (e.g., 2).
- If two levels are relevant, respond with both levels, separated by a comma, in order of relevance (e.g., 1,3A).
- If three levels are relevant, respond with all three levels in order of their suitability (e.g., 1,2,VUS).

Important: Your response must be strictly based on the details and context of each query. Do not default to specific levels unless they are genuinely the most relevant to the query context. Provide a unique and tailored response for each query based on its specific details.

Level of Evidence:

1: FDA-recognized biomarker predictive of response to an FDA-approved drug in this indication

2: Standard care biomarker recommended by the NCCN or other professional guidelines predictive of response to an FDA-approved drug in this indication

3A: Compelling clinical evidence supports the biomarker as being predictive of response to a drug in this indication

3B: Standard care or investigational biomarker predictive of response to an FDA-approved or investigational drug in another indication

4: Compelling biological evidence supports the biomarker as being predictive of response to a drug

R1: Standard care biomarker predictive of resistance to an FDA-approved drug in this indication

R2: Compelling clinical evidence supports the biomarker as being predictive of resistance to a drug

VUS: Variant of unknown clinical significance; no convincing published evidence of cancer association
"""

# Batch size
batch_size = 12
epochs = 10  # Set the number of epochs to run

# List of models to test
models = ["llama3.1:70b_tmp_0"]

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

                # If the model is gpt-4, add family
                if model == "gpt-4":
                    payload["family"] = "Azure OpenAI"

                # Print the current request being processed
                print(f"Model {model} - Epoch {epoch} - Sending request {batch_start + i + 1}/{len(queries)}: {query}")

                # Send POST request to API
                response = requests.post(url, headers=headers, data=json.dumps(payload))
                
                if response.status_code == 200:
                    try:
                        response_data = json.loads(response.text)
                        levels = response_data.get("message", {}).get("content", "").strip()
                        # Print the model's returned value to console
                        print(f"Model {model} - Epoch {epoch} - Response: {levels}")
                        batch_responses.append(levels)
                    except json.JSONDecodeError:
                        batch_responses.append("Response decoding error")
                        print(f"Model {model} - Epoch {epoch} - Query {batch_start + i + 1} response decoding error")
                else:
                    error_msg = f"Error: {response.status_code}"
                    batch_responses.append(error_msg)
                    print(f"Model {model} - Epoch {epoch} - Request {batch_start + i + 1} failed, status code: {response.status_code}")
            
            # Append batch responses to total responses list
            responses.extend(batch_responses)

        # Parse LLM responses and store results in DataFrame
        max_answers = 3  # Assume a maximum of three answers
        for idx, response in enumerate(responses):
            levels = [level.strip() for level in response.split(',') if level.strip()] if isinstance(response, str) else ["N/A"]
            # Handle cases where 1 to 3 answers are returned, filling with "N/A" if necessary
            for answer_idx in range(max_answers):
                col_name = f"Epoch{epoch}_Level{answer_idx + 1}"
                results_df.loc[idx, f"{model.replace(':', '_')}_{col_name}"] = levels[answer_idx] if answer_idx < len(levels) else "N/A"

    # Save results as a separate Excel file
    output_file_path = f'/path/gene_onco/LLM_{model.replace(":", "_")}_all_epochs_response.xlsx'
    results_df.to_excel(output_file_path, index=False)

    print(f"Results for model {model} have been saved to: {output_file_path}")

# Calculate and display total execution time
end_time = time.time()
total_time = end_time - start_time
print(f"Total execution time: {total_time:.2f} seconds")
