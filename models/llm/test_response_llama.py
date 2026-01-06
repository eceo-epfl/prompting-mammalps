from transformers import AutoModelForCausalLM, AutoTokenizer
import re

device = "cuda"

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
).to(device)

with open("function_headers_and_return_types.txt", "r") as f:
    tools = " ".join([s.strip() for s in f.read().splitlines()])

instructions = "Given the available <|tools|>, and the following <|prompt|>, implement the python <|function|> which returns True if the individual animal tracks contains information that fully matches the prompt, and False otherwise. We expect a sequence of function calls from the <|tools|>. Only use the provided functions. Think carefully how to decompose the prompt, and output your final answer as the implementation of the python function."
prompt = "A video of a wolf."
context = f"<|prompt_start|>{prompt}<|prompt_end|><|function_start|>def check_tracks(individual_tracks) -> bool'<|function_end|><|tools_start|>{str(tools)}<|tools_end|>"

messages = [{"role": "user", "content": instructions + context},]

# format and tokenize the tool use prompt 

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
print(text)

model_inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(model.device)

outputs = model.generate(**model_inputs, max_new_tokens = 1000)
print(tokenizer.decode(outputs[0][len(model_inputs.input_ids[0]) :], skip_special_tokens=True))