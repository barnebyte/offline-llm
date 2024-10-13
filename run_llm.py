import os
import re
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    StoppingCriteria,
    StoppingCriteriaList,
)
import torch
import torch.nn.functional as F

# Enable offline mode to prevent internet access
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Set the cache directory to the local model folders
os.environ["TRANSFORMERS_CACHE"] = "/app/models/"

# Load the GPT-J model from the local path
model = AutoModelForCausalLM.from_pretrained(
    "/app/models/gpt-j-6B",
    load_in_8bit=True,  # You can set this to True if you have more than ~10gigs of vram
    device_map="auto",
    local_files_only=True,  # Ensure loading from local files
)
tokenizer = AutoTokenizer.from_pretrained(
    "/app/models/gpt-j-6B",
    local_files_only=True,
)

# Load the BERT model for sequence classification from the local path
bert_model = AutoModelForSequenceClassification.from_pretrained(
    "/app/models/bert-base-uncased-SST-2",
    local_files_only=True,
)
bert_tokenizer = AutoTokenizer.from_pretrained(
    "/app/models/bert-base-uncased-SST-2",
    local_files_only=True,
)

# Set the padding token to eos_token if it's not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def summarize_conversation(conversation_history, max_summary_length=150):
    """
    Summarizes the conversation history using the model.
    """
    conversation_text = "\n".join(conversation_history)
    prompt = (
        "Summarize the following conversation between a user and an assistant in a concise manner.\n\n"
        f"Conversation:\n{conversation_text}\n\nSummary:"
    )

    # Tokenize the prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    # Move inputs to the same device as the model
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    # Generate summary
    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=max_summary_length,
        num_beams=4,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )

    # Decode the summary
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # Extract the summary after 'Summary:'
    summary = summary_text.split("Summary:")[-1].strip()
    # Remove any extra tokens
    summary = summary.split("Assistant:")[0].strip()
    return summary

def generate_response(prompt, conversation_history=None, max_history=4, num_responses=5):
    # Initialize conversation history if not provided
    if conversation_history is None:
        conversation_history = []
        summary = ""
    else:
        summary = ""

    # Append the new user prompt to the conversation history
    conversation_history.append(f"User: {prompt}")

    # Manage conversation history length
    if len(conversation_history) > max_history * 2:  # Each turn includes User and Assistant
        # Summarize the earlier part of the conversation
        summary = summarize_conversation(conversation_history[:-max_history * 2])
        # Keep only the last max_history exchanges
        conversation_history = conversation_history[-max_history * 2:]

    # Construct the prompt with system instructions and conversation
    system_prompt = "You are a helpful assistant."
    if summary:
        system_prompt += f" Here is a summary of the conversation so far: {summary}"

    # Combine system prompt and conversation history
    prompt_text = (
        f"{system_prompt}\n\n"
        + "\n".join(conversation_history)
        + "\nAssistant:"
    )

    # Tokenize the prompt
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048 - 256,  # Reserve tokens for the response
    )

    # Move inputs to the same device as the model
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    # Adjust the stopping criteria to use a unique stop token
    stop_token = "<|endofassistant|>"
    stop_token_id = tokenizer.encode(stop_token, add_special_tokens=False)

    class StopOnTokens(StoppingCriteria):
        def __init__(self, stop_ids):
            super().__init__()
            self.stop_ids = stop_ids

        def __call__(self, input_ids, scores, **kwargs):
            for stop_id in self.stop_ids:
                if input_ids.shape[1] >= len(stop_id):
                    if torch.all(
                        input_ids[0, -len(stop_id) :] == torch.tensor(stop_id).to(input_ids.device)
                    ):
                        return True
            return False

    # Generate multiple responses
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=256,  # Adjusted for longer responses
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,  # Adjusted for better response quality
        repetition_penalty=1.2,  # Discourage repetition
        num_return_sequences=num_responses,  # Generate multiple responses
        stopping_criteria=StoppingCriteriaList([StopOnTokens([stop_token_id])]),
    )

    responses = []
    for output in outputs:
        # Decode and clean up the assistant's response
        generated_text = tokenizer.decode(output, skip_special_tokens=True)

        # Extract the assistant's response
        pattern = r"Assistant:(.*?)(<\|endofassistant\|>|$)"
        match = re.search(pattern, generated_text, re.DOTALL)
        if match:
            assistant_response = match.group(1).strip()
        else:
            assistant_response = generated_text.strip()

        # Remove any instances of 'User:' or conversation history from the assistant's response
        assistant_response = re.sub(r"User:.*", "", assistant_response, flags=re.DOTALL).strip()

        responses.append(assistant_response)

    # Use BERT to select the best response
    best_response = select_best_response(prompt, responses)

    # Append the assistant's response to the conversation history
    conversation_history.append(f"Assistant: {best_response}")

    return best_response, conversation_history

def select_best_response(user_prompt, responses):
    """
    Uses BERT to select the best response from a list of responses.
    """
    # Prepare inputs for BERT
    inputs = [user_prompt + " [SEP] " + response for response in responses]
    bert_inputs = bert_tokenizer(
        inputs,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    # Move inputs to the same device as the BERT model
    bert_inputs = {key: value.to(bert_model.device) for key, value in bert_inputs.items()}

    # Get BERT's predictions
    with torch.no_grad():
        outputs = bert_model(**bert_inputs)
        logits = outputs.logits

    # Since SST-2 is a binary classification task (positive/negative),
    # we can use the 'positive' class logits as a proxy for response quality
    positive_class_index = 1  # Index of the 'positive' class
    scores = F.softmax(logits, dim=1)[:, positive_class_index]

    # Select the response with the highest score
    best_response_index = torch.argmax(scores).item()
    best_response = responses[best_response_index]

    return best_response

# Example usage
if __name__ == "__main__":
    conversation_history = []
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response, conversation_history = generate_response(user_input, conversation_history)
        print(f"Assistant: {response}")
