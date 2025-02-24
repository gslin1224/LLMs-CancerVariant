import requests
import json
import pandas as pd
import time
import os

# Start timing
start_time = time.time()

# Load the FoundationOne dataset
file_path = '/path/00_FoundationOne_Variants_Summary_for_LLM_Test.xlsx'
data = pd.read_excel(file_path, sheet_name='FoundationOne_Variants_Summary') 
# Select the necessary columns to construct queries
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
            "deploymentName": "gpt-4o",
            "proxy": False
        },
    })
}

# New system message
ocr_text = """
### Instructions:
Answer only with the level letter(s) corresponding to the classification, with no additional text, explanations, reasoning, or context included.

You can provide one or more appropriate levels as the answer, up to a maximum of three. Please arrange them in order of suitability. For example:
- If the context indicates that only one level is most suitable, respond with that level (e.g., A).
- If two levels are relevant, respond with both levels, separated by a comma, in order of relevance (e.g., A,B).
- If three levels are relevant, respond with all three levels in order of their suitability (e.g., A,B,VUS).

Important: Your response must be strictly based on the details and context of each query. Do not default to specific levels unless they are genuinely the most relevant to the query context. Provide a unique and tailored response for each query based on its specific details.

Level of Evidence:

**A**: Validated association. Proven/consensus association in human medicine.  
*Examples*:  
- "AML with mutated NPM1" is a provisional entity in WHO classification of acute myeloid leukemia (AML). This mutation should be tested for in clinical trials and is recommended for testing in patients with cytogenetically normal AML.  
- Validated associations are often in routine clinical practice already or are the subject of major clinical trial efforts.

**B**: Clinical evidence. Clinical trial or other primary patient data supports association.  
*Examples*:  
- BRAF V600E is correlated with poor prognosis in papillary thyroid cancer in a study of 187 patients with PTC and other thyroid diseases.  
- The evidence should be supported by observations in multiple patients. Additional support from functional data is desirable but not required.

**C**: Case study. Individual case reports from clinical journals.  
*Examples*:  
- A single patient with FLT3 over-expression responded to the FLT3 inhibitor sunitinib.  
- The study may have involved a large number of patients, but the statement was supported by only a single patient. In some cases, observations from just a handful of patients (e.g., 2-3) or a single family may also be considered a case study/report.

**D**: Preclinical evidence. In vivo or in vitro models support association.  
*Examples*:  
- Experiments showed that AG1296 is effective in triggering apoptosis in cells with the FLT3 internal tandem duplication.  
- The study may have involved some patient data, but support for this statement was limited to in vivo or in vitro models (e.g., mouse studies, cell lines, molecular assays, etc.).

**E**: Inferential association. Indirect evidence.  
*Examples*:  
- CD33 and CD123 expression were significantly increased in patients with NPM1 mutation with FLT3-ITD, indicating these patients may respond to combined anti-CD33 and anti-CD123 therapy.  
- The assertion is at least one step removed from a direct association between a molecular profile (variant) and clinical relevance.

**VUS**: Variant of unknown clinical significance. No convincing published evidence of cancer association.
"""

# Batch size
batch_size = 16
total_epochs = 100  
log_interval = 5  
models = ["gpt-4", "qwen2.5:72b", "llama3.1:70b"]  # Using the original model list

# Log file directory
log_dir = "/path/FoundationOneCivic"
os.makedirs(log_dir, exist_ok=True)  # Ensure directory exists

# Iterate through each model
for model in models:
    print(f"Testing model: {model}")

    # Create DataFrame to store all results
    results_df = pd.DataFrame({
        'Query': queries,
        'Original_Level': original_levels
    })

    # Get total number of queries
    total_questions = len(queries)

    # Iterate through epochs, processing log_interval epochs at a time
    for epoch_start in range(1, total_epochs + 1, log_interval):
        epoch_end = min(epoch_start + log_interval - 1, total_epochs)
        log_file_path = os.path.join(log_dir, f"foundation_LLM_log_{model}_epoch{epoch_start}-{epoch_end}.log")
        
        # Initialize log file
        with open(log_file_path, 'w') as log_file:
            log_file.write(f"Model: {model}, Epochs: {epoch_start}-{epoch_end}\n")
            log_file.write(f"Total Questions: {total_questions}\n")
            log_file.write("-" * 50 + "\n")

        # Print current progress
        print(f"Model {model} - Running Epoch {epoch_start} to {epoch_end}")

        # Process each epoch in the current range
        for epoch in range(epoch_start, epoch_end + 1):
            responses = []

            # Process queries in batches
            for batch_start in range(0, len(queries), batch_size):
                batch_queries = queries[batch_start:batch_start + batch_size]  # Process batch_size queries at a time
                batch_responses = []
                for i, query in enumerate(batch_queries):
                    question_number = batch_start + i + 1  # Get current question number
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
                            levels = response_data.get("message", {}).get("content", "").strip()
                            batch_responses.append(levels)
                            with open(log_file_path, 'a') as log_file:
                                log_file.write(f"[Epoch {epoch}] Question #{question_number}/{total_questions} Query: {query}\nResponse: {levels}\n\n")
                        else:
                            error_msg = f"Error: {response.status_code}"
                            batch_responses.append(error_msg)
                    except Exception as e:
                        batch_responses.append(f"Request failed: {e}")

                responses.extend(batch_responses)

            # Process responses and store in DataFrame
            max_answers = 3
            for idx, response in enumerate(responses):
                levels = [level.strip() for level in response.split(',') if level.strip()] if isinstance(response, str) else ["N/A"]
                for answer_idx in range(max_answers):
                    col_name = f"Epoch{epoch}_Level{answer_idx + 1}"
                    results_df.loc[idx, f"{model.replace(':', '_')}_{col_name}"] = levels[answer_idx] if answer_idx < len(levels) else "N/A"

    # Save results
    output_file_path = os.path.join(log_dir, f"foundation_LLM_{model.replace(':', '_')}.xlsx")
    results_df.to_excel(output_file_path, index=False)
    print(f"Results for model {model} saved to: {output_file_path}")

# Calculate total execution time
end_time = time.time()
total_time = end_time - start_time
print(f"Total execution time: {total_time:.2f} seconds")
