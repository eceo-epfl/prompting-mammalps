import json
from argparse import ArgumentParser
from pathlib import Path
import yaml
import logging
import inspect

import yaml
from parse_json_tools import (
    get_adult_deer_tracks_from_sex,
    get_deer_tracks_from_age,
    get_nb_adult_deer_tracks_sex_in_video,
    get_nb_deer_tracks_age_in_video,
    get_nb_tracks_action_in_video,
    get_nb_tracks_activity_in_video,
    get_nb_tracks_species_in_video,
    get_tracks_from_action,
    get_tracks_from_activity,
    get_tracks_from_json,
    get_tracks_from_species,
    tracks_contain_action,
    tracks_contain_activity,
    tracks_contain_adult_deer_sex,
    tracks_contain_deer_age,
    Species, Action, Activity, DAge, DSex, Meteo
)
from smolagents import CodeAgent, PromptTemplates, TransformersModel


def main(args):
    tools = [
        get_tracks_from_json,
        get_tracks_from_species,
        get_tracks_from_action,
        get_tracks_from_activity,
        get_adult_deer_tracks_from_sex,
        get_deer_tracks_from_age,
        tracks_contain_action,
        tracks_contain_activity,
        tracks_contain_adult_deer_sex,
        tracks_contain_deer_age,
        get_nb_adult_deer_tracks_sex_in_video,
        get_nb_deer_tracks_age_in_video,
        get_nb_tracks_action_in_video,
        get_nb_tracks_activity_in_video,
        get_nb_tracks_species_in_video,
    ]

    with open("prompt.yaml", "r") as f:
        prompts = yaml.load(f, Loader=yaml.SafeLoader)

    prompt_templates = PromptTemplates(prompts)

    # Load model and code agent
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model = TransformersModel(model_id, device_map="cuda")
    agent = CodeAgent(tools=tools, model=model, prompt_templates=prompt_templates, max_print_outputs_length=500)

    agent.python_executor.send_variables({"Species": Species, 
                                          "Activity": Activity,
                                          "Action": Action, 
                                          "DAge": DAge, 
                                          "DSex": DSex, 
                                          "Meteo": Meteo})

    # Logger
    output_log = Path(args.output_folder) / "process.log"
    logging.basicConfig(filename=output_log, filemode="w", 
                                 format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S',
                                level=logging.INFO)
    logger = logging.getLogger()

    # Load queries
    with open(args.input_queries_videos, "r") as f:
        queries_dict = json.load(f)

    queries_list = [q for q_cat in queries_dict.values() for q in q_cat if "<vid>" not in q]
    output_queries_videos = {q: [] for q in queries_list}
    test_file = "./S1_C1_E16_V0040.json"

    # For every query
    for query in queries_list:
        logger.info(f"Processing query {query}")
        message = "Verify if the content of the file matches the following prompt (return True or False):" + f"'{query}'. Don't forget: always match elements from the prompt to the label space; save your implementation of the check_file function first as you will need it again."
        agent.run(message, return_full_result=True, max_steps=5, additional_args={"json_file": test_file})
        
        # Get the function that was created and apply it to all files
        try:
            check_file = agent.python_executor.custom_tools["check_file"]
            check_file_str = inspect.getsource(check_file)
            logging.info(agent.memory.return_full_code())
            output_queries_videos[query] = [f.stem for f in Path(args.json_folder).rglob("*/*.json") if check_file(f)]
        except Exception as e:
            logging.warning(f"Could not apply check_file function for query: {query}")
            logging.warning(e)

        # Save results at every step
        output_json_file = Path(args.output_folder) / (model_id.split("/")[1] + "_retrieval_results.json")
        with open(output_json_file, "w") as f:
            json.dump(output_queries_videos, f, indent=2)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-IQ",
        "--input_queries_videos",
        help="JSON file containing the queries of interest and associated ground truth videos",
    )
    parser.add_argument(
        "-IJ",
        "--json_folder",
        help="Folder containing video prediction or annotation files as structured JSON",
    )
    # parser.add_argument(
    #     "--llm", choices=["apertus", "mistral", "qwen", "llama"], default="llama"
    # )
    parser.add_argument(
        "-O",
        "--output_folder",
        help="output folder containing logs, intermediary results and output files",
    )

    args = parser.parse_args()

    main(args)
