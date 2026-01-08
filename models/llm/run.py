import json
from argparse import ArgumentParser
from pathlib import Path

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
)
from smolagents import CodeAgent, TransformersModel


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

    # Load model and code agent
    instructions = ("Task: Verify if the content of the file matches the prompt."
            "To achieve this, proceed in two steps. First step: define a function named check_file to check if the file contains elements that fully matches the prompt or not. "
            "Only use the provided tools to use in your check_file function. Never 'print' the content of any variable." \
            "Second step: call the newly defined check_file function with the json_file passed as a variable and output the boolean answer."
            "Here are two examples of check_file function definitions."
            "1) 'A video of a roe deer grazing.':  "
            "def check_file(json_file):"
            "   individual_tracks = get_tracks_from_json(json_file)"
            "   roe_deer_tracks = get_tracks_from_species(individual_tracks, species_name='roe deer')"
            "   return tracks_contain_action(roe_deer_tracks, action_name='grazing')"
            "2) 'A video of an animal bathing.':  "
            "def check_file(json_file):"
            "   individual_tracks = get_tracks_from_json(json_file)"
            "   return tracks_contain_action(individual_tracks, action_name='bathing')"
        )
    model = TransformersModel("meta-llama/Meta-Llama-3.1-8B-Instruct", device_map="cuda")
    agent = CodeAgent(tools=tools, model=model, instructions=instructions, max_print_outputs_length=500)
    
    # Load queries
    with open(args.input_queries_videos, "r") as f:
        queries_dict = json.load(f)

    queries_list = [q for q_cat in queries_dict.values() for q in q_cat]
    output_queries_videos = {q: [] for q in queries_list}
    test_file = "./S1_C1_E4_V0016.json"

    # For every query
    for query in queries_list[3:]:
        message = "Verify if the content of the file matches the following prompt (return True or False):" + f"'{query}'. Don't forget to save the implementation of the check_file function first as you will need it again."
        agent.run(message, return_full_result=True, max_steps=5, additional_args={"json_file": test_file})
        
        # Get the function that was created and apply it to all files
        try:
            check_file = agent.python_executor.custom_tools["check_file"]
            output_queries_videos[query] = [f.stem for f in Path(args.json_folder).rglob("*/*.json") if check_file(f)]
        except Exception as e:
            print(f"Could not apply check_file function for query: {query}")
            print(e)

        # Save results at every step
        with open(args.output_queries_videos, "w") as f:
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
    parser.add_argument(
        "--llm", choices=["apertus", "mistral", "qwen", "llama"], default="llama"
    )
    parser.add_argument(
        "-OQ",
        "--output_queries_videos",
        help="JSON file containing retrieved videos for each query",
    )

    args = parser.parse_args()

    main(args)
