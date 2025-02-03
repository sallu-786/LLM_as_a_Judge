from tqdm import tqdm
from utils.build_prompt import build_evaluation_prompt
from litellm import completion
LLAMA="ollama/llama3.1"

def generate_response(prompt):
    """
    Generate a response from Ollama using LiteLLM.
    """
    try:
        response = completion(
            model=LLAMA, #/ChatGPT-4o-mini
            messages=[{"content": prompt, "role": "user"}], 
            # api_base="http://localhost:11434"
        )
        return response.choices[0].message.content if response.choices else None
    except Exception as e:
        print(f"Error generating response from Ollama: {e}")
        return None


def evaluate(prompt):
    try:
        # Get the response from Ollama using LiteLLM
        response = generate_response(prompt)
        return response

    except Exception as e:
        print(f"Error in evaluate function: {e}")
        return None


def parse_response(response):
    """
    Parse model response to extract reasoning and score.
    """
    try:
        # Split into lines and clean up
        lines = [line.strip() for line in response.split('\n') if line.strip()]

        critique = None
        score = None

        for i, line in enumerate(lines):
            if line.startswith("**Reasoning:**"):
                critique = lines[i].replace("**Reasoning:**", "").strip()
            elif line.startswith("**Result:**"):
                score = lines[i].replace("**Result:**", "").strip()

        # Remove style tag if present
        if critique and "<userStyle>" in critique:
            critique = critique.split("<userStyle>")[0].strip()

        return critique, score

    except Exception as e:
        print(f"Error parsing Judge response: {e}")
        return None, None


def evaluate_df(df, prompt, assistant_response_col, user_input_col,
                ground_truth_col=None, output_critique_col='judge_critique',
                output_score_col='judge_score'):
    """
    Evaluate each row in the DataFrame using the specified model and prompt template.
    """
    required_columns = set([assistant_response_col, user_input_col])
    if not(required_columns.issubset(set(df.columns))):
        raise ValueError(f"Columns not found in dataframe: {required_columns - set(df.columns)}")

    df_evaluated = df.copy()
    df_evaluated[output_critique_col] = ''
    df_evaluated[output_score_col] = None

    for index in tqdm(range(len(df)), desc="Evaluating responses", unit="row"):
        try:
            evaluation_prompt = build_evaluation_prompt(
                df,
                prompt=prompt,
                assistant_response_col=assistant_response_col,
                user_input_col=user_input_col,
                ground_truth_col=ground_truth_col,
                row_index=index
            )

            response = evaluate(evaluation_prompt)
            critique, score = parse_response(response)

            df_evaluated.at[index, output_critique_col] = critique
            df_evaluated.at[index, output_score_col] = score

        except Exception as e:
            print(f"Error processing row {index+1}: {e}")
            df_evaluated.at[index, output_critique_col] = f"Error: {str(e)}"
            df_evaluated.at[index, output_score_col] = None

    return df_evaluated
