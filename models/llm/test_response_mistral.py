from transformers import AutoModelForCausalLM, AutoTokenizer
import importlib.util
import inspect
import os
import json

device = "cuda"

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
).to(device)

with open("parse_json_schema_reduced_tracks.json", "r") as f:
    tools = json.load(f)

instructions = "Given the [AVAILABLE_TOOLS], and the following <prompt>, implement the python [FUNCTION] which returns True if the individual animal tracks contains information that fully matches the prompt, and False otherwise. We expect a sequence of function calls from the [AVAILABLE_TOOLS]. Only use the provided functions. Only output the python [FUNCTION] implementation."
prompt = "A video of a wolf."
context = f"[PROMPT]{prompt}[/PROMPT][FUNCTION]'def check_tracks(individual_tracks) -> bool'[/FUNCTION]"

messages = [{"role": "user", "content": instructions + context},]

# format and tokenize the tool use prompt 

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    tools=tools,
)
print(text)

model_inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(model.device)

outputs = model.generate(**model_inputs, max_new_tokens=32000)
print(tokenizer.decode(outputs[0][len(model_inputs.input_ids[0]) :], skip_special_tokens=True))