from gpt import GPT
from pathlib import Path
import pandas as pd
import random
import time
import argparse
from transformers import AutoTokenizer

from datasets import load_dataset, Dataset
from huggingface_hub import create_repo
import torch
from tqdm import tqdm

template = r"""
Which of the following summaries does a better job of summarizing the most important points in the given forum post, without including unimportant or irrelevant details? Judge based on accuracy, coverage, and coherence.

### Post:
{{post}}

### Summary A:
{{summarya}}

### Summary B:
{{summaryb}}

### Instructions:
FIRST provide a one-sentence comparison of the two summaries, explaining which \
you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your choice. Your response should use the format:
Comparison: <one-sentence comparison and explanation>
Preferred: <"A" or "B">
"""

def parse_args():
    parser = argparse.ArgumentParser(description="winrate")
    parser.add_argument('--file_name', type=str, default='iterative_dpo_2_8b_555134.csv')
    return parser.parse_args()

def process_text(post, summary_a, summary_b):
    # print(summary_a)
    # print(summary_b)
    text = template.replace("{{post}}", post)
    text = text.replace("{{summarya}}", summary_a)
    text = text.replace("{{summaryb}}", summary_b)
    return text

def process_response(text, response, i):
    comparison = response.split("Comparison:")[1].split("Preferred:")[0].strip()
    preferred = response.split("Preferred:")[1].strip()
    return comparison, preferred, i, text + response

def winrate(file, file_name, n_samples=64):
    tokenizer = AutoTokenizer.from_pretrained("./models/sft_tldr_pythia_1.4b", padding_side="right", trust_remote_code=True, add_eos_token=True)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    left_tokenizer = AutoTokenizer.from_pretrained("./models/sft_tldr_pythia_1.4b", padding_side="left", trust_remote_code=True, add_eos_token=False)
    left_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    gpt = GPT(model_name="gpt-4o")
    df = pd.read_csv(file, converters={'postprocessed_responses': pd.eval})
    print("Total Comparisons: ", len(df))
    all_responses = df["postprocessed_responses"].tolist()
    # print(all_responses[:1])

    df["shuffled_index"] = [None for _ in range(len(df))]
    df["preferred"] = [None for _ in range(len(df))]
    df["choice"] = [-1 for _ in range(len(df))]
    # df["response0"] = [None for _ in range(len(df))]
    # df["response1"] = [None for _ in range(len(df))]
    # df["query_response0"] = [None for _ in range(len(df))]
    # df["query_response1"] = [None for _ in range(len(df))]
    # df["query_response0_token"] = [None for _ in range(len(df))]
    # df["query_response1_token"] = [None for _ in range(len(df))]
    df[f"iter_{1}_best_query_response"] = [[0] for _ in range(len(df))]
    df[f"iter_{1}_worst_query_response"] = [[0] for _ in range(len(df))]
    df[f"iter_{1}_best_mask"] = [[0] for _ in range(len(df))]
    df[f"iter_{1}_worst_mask"] = [[0] for _ in range(len(df))]
    

    
    queries = []
    if len(df) < n_samples:
        print("Generating for all samples")
        n = list(range(len(df)))
    else:
        # n = random.sample(list(range(len(df))), n_samples)
        n = list(range(n_samples))
    for i in tqdm(n):
        post = df["query"].iloc[i].strip()
        # shuffled the index to avoid GPT4's preference bias in the content's order
        shuffled_index = random.randint(0, 1)
        df.at[i, "shuffled_index"] = shuffled_index
        summaries = [
            all_responses[i][0].strip(),
            all_responses[i][1].strip()
        ]
        summary_a = summaries[shuffled_index]
        summary_b = summaries[1 - shuffled_index]
        processed_query = process_text(post, summary_a, summary_b)
        # print(i)
        queries.append(processed_query)
    
    responses = gpt.generate(queries)
    results = [] 
    errors = []
    for (i, query, response) in tqdm(zip(n, queries, responses)):
        print(i)
        try:
            results.append(process_response(query, response, i))
        except:
            print(f"Error in response {i}")
            errors.append(i)
            results.append((None, None, i, None))

    for _, (_, preferred, i, _) in tqdm(enumerate(results)):
        #df.at[i, "explanation"] = comparison
        #df.at[i, "entire_conversation"] = entire_conversation
        if preferred is not None:
            preferred_label = (
                0
                if (df.at[i, "shuffled_index"] == 0 and preferred == "A")
                or (df.at[i, "shuffled_index"] == 1 and preferred == "B")
                else 1
            )
            df.at[i, "choice"] = preferred_label
            # df.at[i, "response0"] = all_responses[i][0]
            # df.at[i, "response1"] = all_responses[i][1]
            # df.at[i, "query_response0"] = df["query"].iloc[i].strip() + all_responses[i][0]
            # df.at[i, "query_response1"] = df["query"].iloc[i].strip() + all_responses[i][1]
            # df.at[i, "query_response0_token"] = tokenizer(df["query_response0"].iloc[i], padding=True, max_length=565)
            # df.at[i, "query_response1_token"] = tokenizer(df["query_response1"].iloc[i], padding=True, max_length=565)

            query_padded = left_tokenizer(df["query"].iloc[i].strip(), padding="max_length", max_length=512)
            query_padded = torch.tensor(query_padded["input_ids"])

            best_response = all_responses[i][preferred_label]
            best_response_padded = tokenizer(best_response, padding="max_length", max_length=53, truncation=True)
            best_response_padded = torch.tensor(best_response_padded["input_ids"])
            worst_response = all_responses[i][1 - preferred_label]
            worst_response_padded = tokenizer(worst_response, padding="max_length", max_length=53, truncation=True)
            worst_response_padded = torch.tensor(worst_response_padded["input_ids"])

            best_query_response_token = torch.cat((query_padded, best_response_padded))
            worst_query_response_token = torch.cat((query_padded, worst_response_padded))
            best_mask = best_query_response_token.clone()
            best_mask[:512] = tokenizer.pad_token_id
            worst_mask = worst_query_response_token.clone()
            worst_mask[:512] = tokenizer.pad_token_id

            # print(best_query_response_token)

            df.at[i,  f"iter_{1}_best_query_response"] = best_query_response_token.tolist()
            df.at[i,  f"iter_{1}_worst_query_response"] = worst_query_response_token.tolist()
            df.at[i,  f"iter_{1}_best_mask"] = best_mask.tolist()
            df.at[i,  f"iter_{1}_worst_mask"] = worst_mask.tolist()
        else:
            df.at[i, "choice"] = -1

    
    value_counts = df["choice"].value_counts()
    print(value_counts)

    winrate = 100 * (value_counts[0] / len(n))
    print(f"Winrate of first sample: {winrate}%")

    labeled = df[df["choice"] != -1]

    labeled.to_csv(Path(args.file_name[:-4] + "_gpt-4o.csv"), index=False)
    print(f"Saved results {args.file_name[:-4]+ '_gpt-4o.csv'}")

    dataset = Dataset.from_pandas(labeled)
    dataset.push_to_hub("gswamy/pythia-1.4B-tldr-gpt-pair-iter-" + str(1))

if __name__ == '__main__':
    start = time.time()
    args = parse_args()
    n_samples = 100000
    file = Path(args.file_name)
    print(args.file_name, n_samples)
    winrate(file, args.file_name, n_samples=n_samples)
    print('total time:', time.time() - start)