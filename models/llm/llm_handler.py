from transformers import AutoModelForCausalLM, AutoTokenizer

llm_name_to_id = {
    "apertus": "swiss-ai/Apertus-8B-Instruct-2509",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen": "Qwen/Qwen3-8B",
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}


class LLMHandler:
    def __init__(
        self, llm_name: str, device: str = "cuda", max_new_tokens: int = 10000
    ):
        self.llm_name = llm_name
        self.llm_id = llm_name_to_id[llm_name]
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.instructions = (
            "Given the available <|tools|>, "
            "and the following <|prompt|>, "
            "implement the python <|function|> which returns True if the individual animal tracks contains information that fully matches the prompt and False otherwise. "
            "Specifically, return a sequence of function calls from the <|tools|>. "
            "Only use the provided functions. Output your final answer at the end."
        )

        self.messages = None

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_id,
        ).to(self.device)

    def make_instruction_prompt(self, tools_txt):
        self.instructions = self.instructions
        self.tools_txt = tools_txt

    def create_message(self, prompt):
        context = f"<|prompt_start|>{prompt}<|prompt_end|><|function_start|>def check_tracks(individual_tracks) -> bool: is_matching = False #TODO return is_matching'<|function_end|><|tools_start|>{str(self.tools_txt)}<|tools_end|>"
        self.message = [
            {"role": "user", "content": self.instructions + context},
        ]

    def get_llm_response(self):
        text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer(
            [text], return_tensors="pt", add_special_tokens=False
        ).to(self.device)

        outputs_pt = self.model.generate(
            **model_inputs, max_new_tokens=self.max_new_tokens
        )
        outputs_txt = self.tokenizer.decode(
            outputs_pt[0][len(model_inputs.input_ids[0]) :], skip_special_tokens=True
        )

        self.outputs_txt = outputs_txt
