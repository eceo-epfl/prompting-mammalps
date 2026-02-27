import json
import logging
import os
from argparse import ArgumentParser
from pathlib import Path

import yaml
from parse_json_tools import (
    Action,
    Activity,
    DAge,
    DSex,
    Meteo,
    Species,
    get_weather_condition_from_file_id,
    check_contains_weather_condition,
    check_track_contains_continuous_sequence,
    get_action_sequences_from_tracks,
    get_adult_deer_tracks_from_sex,
    get_deer_tracks_from_age,
    get_nb_adult_deer_tracks_sex_in_video,
    get_nb_deer_tracks_age_in_video,
    get_nb_tracks_action_in_video,
    get_nb_tracks_activity_in_video,
    get_nb_tracks_species_in_video,
    get_tracks_from_action,
    get_tracks_from_activity,
    get_tracks_from_file_id,
    get_tracks_from_species,
    get_unique_actions_from_tracks,
    get_unique_activities_from_tracks,
    get_unique_species_from_tracks,
    check_tracks_contain_action,
    check_tracks_contain_activity,
    check_tracks_contain_adult_deer_sex,
    check_tracks_contain_deer_age,
    check_tracks_contain_species,
)
from smolagents import CodeAgent, PromptTemplates, TransformersModel


def main(args):
    tools = [
        get_tracks_from_file_id,
        get_weather_condition_from_file_id,
        get_tracks_from_action,
        get_tracks_from_activity,
        get_tracks_from_species,
        get_unique_actions_from_tracks,
        get_unique_activities_from_tracks,
        get_unique_species_from_tracks,
        get_action_sequences_from_tracks,
        get_adult_deer_tracks_from_sex,
        get_deer_tracks_from_age,
        get_nb_adult_deer_tracks_sex_in_video,
        get_nb_deer_tracks_age_in_video,
        get_nb_tracks_action_in_video,
        get_nb_tracks_activity_in_video,
        get_nb_tracks_species_in_video,
        check_contains_weather_condition,
        check_track_contains_continuous_sequence,
        check_tracks_contain_action,
        check_tracks_contain_activity,
        check_tracks_contain_adult_deer_sex,
        check_tracks_contain_deer_age,
        check_tracks_contain_species,
    ]

    with open("prompt.yaml", "r") as f:
        prompts = yaml.load(f, Loader=yaml.SafeLoader)

    prompt_templates = PromptTemplates(prompts)

    # Load model and code agent
    if args.llm == "qwen":
        model_id = "Qwen/Qwen3-8B"  # "Qwen/Qwen3-Coder-Next"  # "meta-llama/Meta-Llama-3.1-8B-Instruct"
    elif args.llm == "llama":
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    elif args.llm == "mistral":
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
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
    queries_list = [q for (cat, q_cat) in queries_dict.items() for q in q_cat if cat=="VIDEO_COMPARISON"]
    output_queries_functions = {}
    test_file_id = "S1_C1_E57_V0141"

    # For every query
    for query in queries_list:  # [31:]:
        logger.info(f"################ Processing query {query} ################")
        message = (
            "Verify if the content of the json file id matches the following prompt (return True or False):"
            + f"'{query}'. Don't forget: always match elements from the prompt to the label space; save your implementation of the check_file function first as you will need it again."
        )
        agent.run(
            message,
            return_full_result=True,
            max_steps=10,
            additional_args={"file_id": test_file_id},
        )

        # Get the function that was created and apply it to all files
        try:
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
        output_json_file = Path(args.output_folder) / (
            model_id.split("/")[1] + "_generated_functions.json"
        )
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

    args = parser.parse_args()

    main(args)
