import json
RAG_hallucination_prompt = """You are tasked with evaluating a response based on a given user input and binary scoring rubric that serves as the evaluation standard. Provide comprehensive feedback on the response quality strictly adhering to the scoring rubric, followed by a binary Yes/No judgment. Avoid generating any additional opening, closing, or explanations.

  Here are some rules of the evaluation:
  (1) You should prioritize evaluating whether the response satisfies the provided rubric. The basis of your score should depend exactly on the rubric. However, the response does not need to explicitly address points raised in the rubric. Rather, evaluate the response based on the criteria outlined in the rubric.

  Your reply should strictly follow this format:
  **Reasoning:** <Your feedback>

  **Result:** <YES or NO>
  Here is the data:

  Instruction:
  ```
  {user_input}
  ```

  Response:
  ```
  {assistant_response}
  ```

  Score Rubrics:
  Evaluate whether the information provided in the response is factually accurate and directly supported by the context given in the related passages.
  Yes: The response is factually accurate and directly supported by the information provided in the passages, without any fabricated or hallucinated details.
  No: The response contains any information that is not supported by the passages, includes fabricated details, or misinterprets the information from the passages."""

# """### Define function to format prompt with input variables"""



def build_evaluation_prompt(df, user_input_col, assistant_response_col, prompt=RAG_hallucination_prompt, row_index=0, ground_truth_col=None):

    if assistant_response_col not in df.columns:
        raise ValueError(f"Missing required column: {assistant_response_col}")

    if user_input_col not in df.columns:
        raise ValueError(f"Missing required column: {user_input_col}")

    if row_index >= len(df):
        raise ValueError(f"Row index {row_index} is out of bounds")

    # Extract response and input
    assistant_response = df[assistant_response_col].iloc[row_index]
    user_input = df[user_input_col].iloc[row_index]

    # Extract ground truth if column is provided
    ground_truth = ""
    if ground_truth_col and ground_truth_col in df.columns:     #check if ground truth col is not empty or None and is a column innot just an entry
        ground_truth = df[ground_truth_col].iloc[row_index]


    # Parse source_info (if JSON-like) for user_input to avoid errors
    if isinstance(user_input, str) and user_input.startswith("{"):
        try:
            user_input = json.loads(user_input).get("question", user_input)  # Handle missing "question"
        except json.JSONDecodeError:
            pass  # If not valid JSON, use as is

    # Format the prompt
    return prompt.format(
        user_input=user_input,
        assistant_response=assistant_response,
        ground_truth=ground_truth  # Include ground truth if required by the prompt
    )