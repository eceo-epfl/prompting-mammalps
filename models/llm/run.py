import json
from argparse import ArgumentParser

from llm_handler import LLMHandler
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
from smolagents import CodeAgent


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

    # Load model
    llm_handler = LLMHandler(args.llm)
    agent = CodeAgent(tools=tools, model=llm_handler.model)

    # Create instruction prompt
    with open(args.library_description, "r") as f:
        tools = " ".join([s.strip() for s in f.read().splitlines()])
    # llm_handler.make_instruction_prompt(tools_txt=tools)

    # Load queries
    with open(args.input_queries_videos, "r") as f:
        queries_dict = json.load(f)

    queries_list = [q for q_cat in queries_dict.values() for q in q_cat]
    output_queries_videos = {q: [] for q in queries_list}

    # For every query
    for query in queries_list:
        print(query)
        # Build prompt = Context + prompt
        # llm_handler.create_message(prompt=query)

        # Encode and process
        # llm_handler.get_llm_response()

        # Transform into callable
        # llm_handler.parse_output()

        # For every JSON file
        # Apply callable to check if file matches prompt or not
        # Update matching list accordingly

    # Save results
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
