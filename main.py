"""This code was inspired by program from  atla-ai 
https://colab.research.google.com/github/atla-ai/selene-mini/blob/main/cookbooks/HF_Quickstart_Hallucination.ipynb
"""

from datasets import load_dataset
from itertools import islice
import pandas as pd
from utils.build_prompt import build_evaluation_prompt
from utils.evaluate import evaluate_df
from utils.plots import pie_chart, dist_plot, conf_mat

# Load the dataset from huggingface (Only the first 100 samples)
RAGTruth_dataset = load_dataset("flowaicom/RAGTruth_test", split="qa", streaming=True)
RAGTruth_dataset = list(islice(RAGTruth_dataset, 0, 100))

# Convert to DataFrame
df = pd.DataFrame(RAGTruth_dataset)
pd.set_option("display.max_colwidth", 500)
pd.set_option('display.max_rows', 500)

# Create the evaluation prompt for the first row of test data
RAG_hallucination_prompt = build_evaluation_prompt(df=df,
                                                   user_input_col='prompt',
                                                   assistant_response_col='response', 
                                                   ground_truth_col='score',
                                                   row_index=0)

# Evaluate the dataset using LLM
df_evaluated = evaluate_df(df=df,
                           prompt=RAG_hallucination_prompt,
                           user_input_col='prompt',
                           assistant_response_col='response',)


print(df_evaluated['judge_score'])
print("----------------------------------------------------------------")
#incase if LLM fails to responed properly remove missing/nan values
df_evaluated = df_evaluated[df_evaluated['judge_score'] != '']  
df_evaluated = df_evaluated.dropna(subset=['judge_score'])  

df_evaluated['judge_score'] = df_evaluated['judge_score'].map({'YES': 1, 'NO': 0})

# Calculate accuracy and draw plots
accuracy = (df_evaluated['judge_score'] == df_evaluated['score']).mean()
print(f"Accuracy: {accuracy:.2%}")

pie_chart(df_evaluated)
dist_plot(df_evaluated)
conf_mat(df_evaluated)