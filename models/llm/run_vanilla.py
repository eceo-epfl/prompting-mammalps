import json
import logging
import os
from argparse import ArgumentParser
from pathlib import Path

import yaml
from parse_json_tools_no_label_constrain import (
    Action,
    Activity,
    DAge,
    DSex,
    Meteo,
    Species,
    load_json_from_id,
)
from smolagents import CodeAgent, PromptTemplates, TransformersModel


def main(args):
    tools = [
        load_json_from_id,
    ]

    with open(args.yaml, "r") as f:
        prompts = yaml.load(f, Loader=yaml.SafeLoader)

    prompt_templates = PromptTemplates(prompts)

    # Load model and code agent
    if args.llm == "qwen":
        model_id = "Qwen/Qwen3-8B"  # "Qwen/Qwen3-Coder-Next"  # "meta-llama/Meta-Llama-3.1-8B-Instruct"
    elif args.llm == "llama":
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    elif args.llm == "mistral":
        model_id = "mistralai/Ministral-3-8B-Instruct-2512"
    elif args.llm == "apertus":
        model_id = "swiss-ai/Apertus-8B-Instruct-2509"
    else:
        raise NotImplementedError()
    model = TransformersModel(model_id, device_map="cuda", max_new_tokens=8096, do_sample=False)
    agent = CodeAgent(
        tools=tools,
        model=model,
        prompt_templates=prompt_templates,
        max_print_outputs_length=500,
    )

    agent.python_executor.send_variables(
        {
            "Species": Species,
            "Activity": Activity,
            "Action": Action,
            "DAge": DAge,
            "DSex": DSex,
            "Meteo": Meteo,
        }
    )


    os.makedirs(args.output_folder, exist_ok=True)
    # Logger
    output_log = Path(args.output_folder) / (model_id.split("/")[1] + "_process.log")
    logging.basicConfig(
        filename=output_log,
        filemode="w",
        format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger()

    # Load queries
    with open(args.input_queries_videos, "r") as f:
        queries_dict = json.load(f)

    # queries_list = [q for q_cat in queries_dict.values() for q in q_cat]
    queries_list = [q for (cat, q_cat) in queries_dict.items() for q in q_cat]
    output_queries_functions = {}
    test_file_id = "S3_C3_E692_V0698"

    # For every query
    output_json_file = Path(args.output_folder) / (
            model_id.split("/")[1] + "_generated_functions.json"
    )
    if os.path.exists(output_json_file):
        with open(output_json_file, "r") as f:
            output_queries_functions = json.load(f)

    for query in queries_list:  # [31:]:
        if query in output_queries_functions:
            print("Skipping already processed query:", query)
            continue
        logger.info(f"################ Processing query {query} ################")
        message = (
            "Verify if the content of the json file id matches the following prompt (return True or False):"
            + f"'{query}'. Don't forget: always match elements from the prompt to the label space; save your implementation of the check_file function first as you will need it again."
        )
        try:
            agent.run(
                message,
                return_full_result=True,
                max_steps=10,
                additional_args={"file_id": test_file_id},
            )

            # Get the function that was created and apply it to all files
            check_file = agent.python_executor.custom_tools["check_file"]
            check_file_str = check_file.__source__
            logging.info(
                f"\t {json.dumps(agent.memory.get_succinct_steps(), indent=2)}"
            )
            output_queries_functions[query] = check_file_str
        except Exception as e:
            logging.warning(f"Could not apply check_file function for query: {query}")
            logging.warning(e)

        # Save results at every step
        with open(output_json_file, "w") as f:
            json.dump(output_queries_functions, f, indent=2)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-IQ",
        "--input_queries_videos",
        help="JSON file containing the queries of interest and associated ground truth videos",
    )
    parser.add_argument("--llm", choices=["qwen", "llama", "mistral", "apertus"], default="llama")
    parser.add_argument(
        "-O",
        "--output_folder",
        help="output folder containing logs, intermediary results and output files",
    )

    parser.add_argument("--yaml", default="prompt.yaml", help="Instruction file")
    args = parser.parse_args()

    main(args)
