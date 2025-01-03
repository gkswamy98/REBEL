from gpt import GPT
from pathlib import Path
import pandas as pd
import random
import time
import argparse
from typing import Optional

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
    parser.add_argument('--n', type=int, default=None)
    return parser.parse_args()

def process_text(post, summary_a, summary_b):
    text = template.replace("{{post}}", post)
    text = text.replace("{{summarya}}", summary_a)
    text = text.replace("{{summaryb}}", summary_b)
    return text

def process_response(text, response, i):
    comparison = response.split("Comparison:")[1].split("Preferred:")[0].strip()
    preferred = response.split("Preferred:")[1].strip()
    return comparison, preferred, i, text + response

def winrate(file, n_samples=64):
    gpt = GPT(model_name="gpt-4o")
    df = pd.read_csv(file)
    df["shuffled_index"] = [None for _ in range(len(df))]
    df["preferred"] = [None for _ in range(len(df))]
    
    queries = []
    if len(df) < n_samples:
        n = list(range(len(df)))
    else:
        n = random.sample(list(range(len(df))), n_samples)
        # n = list(range(n_samples))
    for i in n:
        post = df["query"].iloc[i].strip()
        # shuffled the index to avoid GPT4's preference bias in the content's order
        shuffled_index = random.randint(0, 1)
        df.at[i, "shuffled_index"] = shuffled_index
        summaries = [
            df["postprocessed_response"].iloc[i].strip(),
            df["reference_responses"].iloc[i].split("<|endoftext|>")[0].strip(),
        ]
        summary_a = summaries[shuffled_index]
        summary_b = summaries[1 - shuffled_index]
        processed_query = process_text(post, summary_a, summary_b)
        # print(i)
        queries.append(processed_query)
    
    responses = gpt.generate(queries)
    results = [] 
    for (i, query, response) in zip(n, queries, responses):
        results.append(process_response(query, response, i))

    for _, (comparison, preferred, i, entire_conversation) in enumerate(results):
        #df.at[i, "explanation"] = comparison
        #df.at[i, "entire_conversation"] = entire_conversation
        preferred_label = (
            "ours"
            if (df.at[i, "shuffled_index"] == 0 and preferred == "A")
            or (df.at[i, "shuffled_index"] == 1 and preferred == "B")
            else "reference"
        )
        df.at[i, "preferred"] = preferred_label
    
    value_counts = df["preferred"].value_counts()
    print(value_counts)

    winrate = 100 * (value_counts['ours'] / len(n))
    print(f"Winrate: {winrate}%")
    return winrate

    # df.to_csv(file[:-4] + "_gpt-4o.csv", index=False)
    # print(f"Saved results {file[:-4] + '_gpt-4o.csv'}")

if __name__ == '__main__':
    start = time.time()
    args = parse_args()
    n_samples = 600
    file = Path(args.file_name)
    print(args.file_name, n_samples)
    wr = winrate(file, n_samples=n_samples)
    if args.n is not None:
        print(f"BoN w/ N={args.n}")
        output_file = file.parent / f"winrate_{args.n}.txt"
        with open(output_file, "w") as f:
            f.write(f"Winrate: {wr}%")
        print(f"Saved winrate to {output_file}")
    else:
        output_file = file.parent / f"winrate.txt"
        with open(output_file, "w") as f:
            f.write(f"Winrate: {wr}%")
        print(f"Saved winrate to {output_file}")
    print('total time:', time.time() - start)