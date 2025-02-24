import requests
import json
import pandas as pd
import time
import os

# Start timing
start_time = time.time()

# Read FoundationOne dataset
file_path = '/path/00_FoundationOne_Variants_Summary_for_LLM_Test.xlsx'
data = pd.read_excel(file_path, sheet_name='FoundationOne_Variants_Summary')

# Select necessary columns to construct queries
selected_columns = data[['gene', 'Variant', 'TumorType']]
original_levels = data['Level'].tolist()

# Construct query list
queries = [
    f"Given the gene {row['gene']}, with alteration {row['Variant']} in the context of {row['TumorType']}, what is the appropriate classification?"
    for _, row in selected_columns.iterrows()
]

# Define API details
url = ""
headers = {
    "Content-Type": "application/json",
    "X-chat-ollama-keys": json.dumps({
        "ollama": {"endpoint": "", "username": "", "password": ""},
        "openai": {"key": "", "endpoint": "", "proxy": False},
        "azureOpenai": {
            "key": "",
            "endpoint": "",
            "deploymentName": "gpt-4o",
            "proxy": False
        },
    })
}

# New system message (Binary Classification)
system_prompt = """
System prompts (Binary classification of cancer genetic variants): 

### Role & Objective:
You are an expert assistant specializing in the classification of cancer genetic variants based on clinical significance. Your task is to classify a given gene variant as either:
- Clinically Relevant
- Variant of Unknown Clinical Significance (VUS)


### Scope & Behavior
Only classify gene variants as Clinically Relevant or VUS based on the predefined criteria.
Do not generate explanations, justifications, or interpretations.
Do not provide additional context or assumptions beyond the given query.


### Input & Output Format
Input Format:
You will receive a natural language query specifying the following elements:
Gene Name (e.g., EGFR, KRAS, BRAF)
Alteration (e.g., L858R, G12D, V600E, amplification, fusion)
Tumor Type (e.g., non-small cell lung cancer, colorectal cancer, melanoma)

Example input:
"Given the gene EGFR, with alteration L858R in the context of non-small cell lung cancer, what is the appropriate classification?"


### Output Format:
Return one of the following classifications:
- Clinically Relevant (if the variant meets clinical significance criteria)
- VUS (if the variant is considered a Variant of Unknown Clinical Significance)
No additional text or explanations should be included.

Example Outputs:
Clinically Relevant
VUS


### Clinical Significance Criteria:
A variant is considered Clinically Relevant if it meets at least one of the following criteria:
A: Validated association. 
Proven/consensus association in human medicine.
Examples:
"AML with mutated NPM1" is a provisional entity in WHO classification of acute myeloid leukemia (AML). This mutation should be tested for in clinical trials and is recommended for testing in patients with cytogenetically normal AML. Validated associations are often in routine clinical practice already or are the subject of major clinical trial efforts.

B: Clinical evidence. 
Clinical trial or other primary patient data supports association.
Examples:
BRAF V600E is correlated with poor prognosis in papillary thyroid cancer in a study of 187 patients with PTC and other thyroid diseases. The evidence should be supported by observations in multiple patients. Additional support from functional data is desirable but not required.

C: Case study. 
Individual case reports from clinical journals.
Examples:
A single patient with FLT3 over-expression responded to the FLT3 inhibitor sunitinib. The study may have involved a large number of patients, but the statement was supported by only a single patient. In some cases, observations from just a handful of patients (e.g. 2-3) or a single family may also be considered a case study/report.

D: Preclinical evidence. 
In vivo or in vitro models support association.
Examples:
Experiments showed that AG1296 is effective in triggering apoptosis in cells with the FLT3 internal tandem duplication. The study may have involved some patient data, but support for this statement was limited to in vivo or in vitro models (e.g. mouse studies, cell lines, molecular assays, etc.).

E: Inferential association. 
Indirect evidence.
Examples:
CD33 and CD123 expression were significantly increased in patients with NPM1 mutation with FLT3-ITD, indicating these patients may respond to combined anti-CD33 and anti-CD123 therapy. The assertion is at least one step removed from a direct association between a molecular profile (variant) and clinical relevance.

Handling of VUS:
A variant is classified as VUS (Variant of Unknown Clinical Significance) if:
- It does not meet any of the above criteria (A–E).
- There is insufficient or conflicting evidence regarding its clinical impact.

"""

# Set batch size and epochs
batch_size = 16
total_epochs = 10
log_interval = 5  # Log every few epochs
models = ["qwen2.5:72b"]
log_dir = "/path/FoundationOne_Binary"
os.makedirs(log_dir, exist_ok=True)

# Iterate through each model
for model in models:
    print(f"Testing model: {model}")
    results_df = pd.DataFrame({'Query': queries, 'Original_Level': original_levels})
    total_questions = len(queries)

    for epoch_start in range(1, total_epochs + 1, log_interval):
        epoch_end = min(epoch_start + log_interval - 1, total_epochs)
        log_file_path = os.path.join(log_dir, f"foundation_LLM_log_{model}_epoch{epoch_start}-{epoch_end}.log")

        with open(log_file_path, 'w') as log_file:
            log_file.write(f"Model: {model}, Epochs: {epoch_start}-{epoch_end}\n")
            log_file.write(f"Total Questions: {total_questions}\n")
            log_file.write("-" * 50 + "\n")

        print(f"Model {model} - Executing Epoch {epoch_start} to {epoch_end}")

        for epoch in range(epoch_start, epoch_end + 1):
            responses = []
            for batch_start in range(0, len(queries), batch_size):
                batch_queries = queries[batch_start:batch_start + batch_size]
                batch_responses = []
                
                for i, query in enumerate(batch_queries):
                    question_number = batch_start + i + 1
                    payload = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": query}
                        ]
                    }
                    
                    if model == "gpt-4":
                        payload["family"] = "Azure OpenAI"
                    
                    try:
                        response = requests.post(url, headers=headers, data=json.dumps(payload))
                        if response.status_code == 200:
                            response_data = json.loads(response.text)
                            classification = response_data.get("message", {}).get("content", "").strip()
                            batch_responses.append(classification)
                            with open(log_file_path, 'a') as log_file:
                                log_file.write(f"[Epoch {epoch}] Question #{question_number}/{total_questions} Query: {query}\nResponse: {classification}\n\n")
                        else:
                            error_msg = f"Error: {response.status_code}"
                            batch_responses.append(error_msg)
                            with open(log_file_path, 'a') as log_file:
                                log_file.write(f"[Epoch {epoch}] Question #{question_number}/{total_questions} Query: {query}\nResponse: {error_msg}\n\n")
                    except Exception as e:
                        batch_responses.append(f"Request failed: {e}")
                        with open(log_file_path, 'a') as log_file:
                            log_file.write(f"[Epoch {epoch}] Question #{question_number}/{total_questions} Query: {query}\nResponse: Request failed: {e}\n\n")
                
                responses.extend(batch_responses)
            
            results_df[f"{model.replace(':', '_')}_Epoch{epoch}_Classification"] = responses
    
    output_file_path = os.path.join(log_dir, f"foundation_LLM_binary_{model.replace(':', '_')}.xlsx")
    results_df.to_excel(output_file_path, index=False)
    print(f"Results for model {model} have been saved to: {output_file_path}")

end_time = time.time()
total_time = end_time - start_time
print(f"Total execution time: {total_time:.2f} seconds")
