import argparse
import json

from benchmark.generate_random_queries import Action, Activity, DAge, DSex, Meteo, Species
from parse_json import *


def get_tracks_from_json(json_file):

    video_detections = load_video_detections(json_file)
    individual_tracks = get_tracks_from_video_detections(video_detections)

    return individual_tracks


def get_parsing_function(prompt):
    if prompt == "A video of an animal running.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            return tracks_contain_action(
                individual_tracks, action_name=Action.RUNNING.name.lower()
            )

        return check_file

    elif prompt == "A video of an animal bathing.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            return tracks_contain_action(
                individual_tracks, action_name=Action.BATHING.name.lower()
            )

        return check_file

    elif prompt == "A video of a roe deer grazing.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            roe_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.ROE_DEER.name.lower()
            )
            return tracks_contain_action(
                roe_deer_tracks, action_name=Action.GRAZING.name.lower()
            )

        return check_file

    elif prompt == "A video of an animal browsing.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            return tracks_contain_action(
                individual_tracks, action_name=Action.BROWSING.name.lower()
            )

        return check_file

    elif prompt == "A video of a female adult roe deer sniffing.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            female_roe_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.FEMALE.name.lower(),
                deer_species=Species.ROE_DEER.name.lower(),
            )
            return tracks_contain_action(
                female_roe_deer_tracks, action_name=Action.SNIFFING.name.lower()
            )

        return check_file

    elif prompt == "A video of a juvenile red deer scratching body.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            juvenile_red_deer_tracks = get_deer_tracks_from_age(
                individual_tracks,
                age=DAge.JUVENILE.name.lower(),
                deer_species=Species.RED_DEER.name.lower(),
            )
            return tracks_contain_action(
                juvenile_red_deer_tracks,
                action_name=Action.SCRATCHING_BODY.name.lower(),
            )

        return check_file

    elif prompt == "A video of a juvenile roe deer pushing its mother's udder.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            juvenile_roe_deer_tracks = get_deer_tracks_from_age(
                individual_tracks,
                age=DAge.JUVENILE.name.lower(),
                deer_species=Species.ROE_DEER.name.lower(),
            )
            return tracks_contain_action(
                juvenile_roe_deer_tracks,
                action_name=Action.PUSHING_MOTHERS_UDDER.name.lower(),
            )

        return check_file

    elif prompt == "A video of a chamois resting.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            chamois_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.CHAMOIS.name.lower()
            )
            return tracks_contain_activity(
                chamois_tracks, activity_name=Activity.RESTING.name.lower()
            )

        return check_file

    elif prompt == "A video of a adult red deer reacting to the camera.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            adult_red_deer_tracks = get_deer_tracks_from_age(
                individual_tracks,
                age=DAge.ADULT.name.lower(),
                deer_species=Species.RED_DEER.name.lower(),
            )
            return tracks_contain_activity(
                adult_red_deer_tracks,
                activity_name=Activity.CAMERA_REACTION.name.lower(),
            )

        return check_file

    elif prompt == "A video of a male adult roe deer jumping.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            male_roe_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.MALE.name.lower(),
                deer_species=Species.ROE_DEER.name.lower(),
            )
            return tracks_contain_action(
                male_roe_deer_tracks, action_name=Action.JUMPING.name.lower()
            )

        return check_file

    elif prompt == "A video of a male adult red deer marking with its antlers.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            male_red_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.MALE.name.lower(),
                deer_species=Species.RED_DEER.name.lower(),
            )
            return tracks_contain_action(
                male_red_deer_tracks, action_name=Action.MARKING_ANTLERS.name.lower()
            )

        return check_file

    # COURTSHIP
    elif (
        prompt
        == "A video of a adult red deer standing head up while participating in courtship."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            individual_courtship = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.COURTSHIP.name.lower()
            )
            adult_deer_courtship = get_deer_tracks_from_age(
                individual_courtship,
                age=DAge.ADULT.name.lower(),
                deer_species=Species.RED_DEER.name.lower(),
            )
            return tracks_contain_action(
                adult_deer_courtship, action_name=Action.STANDING_HEAD_UP.name.lower()
            )

        return check_file

    elif (
        prompt
        == "A video of three or more male adult red deer marking with its antlers while participating in courtship."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            individual_courtship = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.COURTSHIP.name.lower()
            )
            male_deer_courtship = get_adult_deer_tracks_from_sex(
                individual_courtship,
                sex=DSex.MALE.name.lower(),
                deer_species=Species.RED_DEER.name.lower(),
            )
            return (
                get_nb_tracks_action_in_video(
                    male_deer_courtship, action_name=Action.MARKING_ANTLERS.name.lower()
                )
                >= 3
            )

        return check_file

    elif (
        prompt
        == "A video of a adult red deer vocalizing while participating in courtship."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            individual_courtship = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.COURTSHIP.name.lower()
            )
            adult_deer_courtship = get_deer_tracks_from_age(
                individual_courtship,
                age=DAge.ADULT.name.lower(),
                deer_species=Species.RED_DEER.name.lower(),
            )
            return tracks_contain_action(
                adult_deer_courtship, action_name=Action.VOCALIZING.name.lower()
            )

        return check_file

    elif (
        prompt
        == "A video of two adult red deer vocalizing while participating in courtship."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            individual_courtship = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.COURTSHIP.name.lower()
            )
            adult_deer_courtship = get_deer_tracks_from_age(
                individual_courtship,
                age=DAge.ADULT.name.lower(),
                deer_species=Species.RED_DEER.name.lower(),
            )
            return (
                get_nb_tracks_action_in_video(
                    adult_deer_courtship, action_name=Action.VOCALIZING.name.lower()
                )
                == 2
            )

        return check_file

    elif (
        prompt == "A video of a adult red deer laying while participating in courtship."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            individual_courtship = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.COURTSHIP.name.lower()
            )
            adult_deer_courtship = get_deer_tracks_from_age(
                individual_courtship,
                age=DAge.ADULT.name.lower(),
                deer_species=Species.RED_DEER.name.lower(),
            )
            return tracks_contain_action(
                adult_deer_courtship, action_name=Action.LAYING.name.lower()
            )

        return check_file

    # SOCIAL
    elif prompt == "A video of three or more red deer.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            return (
                get_nb_tracks_species_in_video(
                    individual_tracks, species_name=Species.RED_DEER.name.lower()
                )
                >= 3
            )

        return check_file

    elif prompt == "A video of two red deer.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            return (
                get_nb_tracks_species_in_video(
                    individual_tracks, species_name=Species.RED_DEER.name.lower()
                )
                == 2
            )

        return check_file

    elif prompt == "A video of three or more red deer reacting to the camera.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            red_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.RED_DEER.name.lower()
            )
            return (
                get_nb_tracks_activity_in_video(
                    red_deer_tracks, activity_name=Activity.CAMERA_REACTION.name.lower()
                )
                >= 3
            )

        return check_file

    elif prompt == "A video of three or more red deer escaping.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            red_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.RED_DEER.name.lower()
            )
            return (
                get_nb_tracks_activity_in_video(
                    red_deer_tracks, activity_name=Activity.ESCAPING.name.lower()
                )
                >= 3
            )

        return check_file

    elif prompt == "A video of a juvenile red deer walking while in vigilance.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            juvenile_red_deer_tracks = get_deer_tracks_from_age(
                individual_tracks,
                age=DAge.JUVENILE.name.lower(),
                deer_species=Species.RED_DEER.name.lower(),
            )
            juvenile_red_deer_tracks_vigilance = get_tracks_from_activity(
                juvenile_red_deer_tracks, activity_name=Activity.VIGILANCE.name.lower()
            )
            return tracks_contain_action(
                juvenile_red_deer_tracks_vigilance,
                action_name=Action.WALKING.name.lower(),
            )

        return check_file

    elif prompt == "A video of two female adult red deer walking while foraging.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            female_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.FEMALE.name.lower(),
                deer_species=Species.RED_DEER.name.lower(),
            )
            female_deer_tracks_foraging = get_tracks_from_activity(
                female_deer_tracks, activity_name=Activity.FORAGING.name.lower()
            )
            return (
                get_nb_tracks_action_in_video(
                    female_deer_tracks_foraging, action_name=Action.WALKING.name.lower()
                )
                == 2
            )

        return check_file

    elif (
        prompt == "A video of three or more male adult red deer walking while foraging."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            male_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.MALE.name.lower(),
                deer_species=Species.RED_DEER.name.lower(),
            )
            male_deer_tracks_foraging = get_tracks_from_activity(
                male_deer_tracks, activity_name=Activity.FORAGING.name.lower()
            )
            return (
                get_nb_tracks_action_in_video(
                    male_deer_tracks_foraging, action_name=Action.WALKING.name.lower()
                )
                >= 3
            )

        return check_file

    # CAMERA_REACTION
    elif prompt == "A video of a adult roe deer reacting to the camera.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            cam_reaction_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION.name.lower()
            )
            return tracks_contain_deer_age(
                cam_reaction_tracks,
                age=DAge.ADULT.name.lower(),
                deer_species=Species.ROE_DEER.name.lower(),
            )

        return check_file

    elif prompt == "A video of a juvenile roe deer reacting to the camera.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            cam_reaction_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION.name.lower()
            )
            return tracks_contain_deer_age(
                cam_reaction_tracks,
                age=DAge.JUVENILE.name.lower(),
                deer_species=Species.ROE_DEER.name.lower(),
            )

        return check_file

    elif prompt == "A video of a male adult red deer reacting to the camera.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            cam_reaction_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION.name.lower()
            )
            return tracks_contain_adult_deer_sex(
                cam_reaction_tracks,
                sex=DSex.MALE.name.lower(),
                deer_species=Species.RED_DEER.name.lower(),
            )

        return check_file

    elif prompt == "A video of a juvenile red deer reacting to the camera.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            cam_reaction_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION.name.lower()
            )
            return tracks_contain_deer_age(
                cam_reaction_tracks,
                age=DAge.JUVENILE.name.lower(),
                deer_species=Species.RED_DEER.name.lower(),
            )

        return check_file

    elif prompt == "A video of a female adult roe deer reacting to the camera.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            cam_reaction_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION.name.lower()
            )
            return tracks_contain_adult_deer_sex(
                cam_reaction_tracks,
                sex=DSex.FEMALE.name.lower(),
                deer_species=Species.ROE_DEER.name.lower(),
            )

        return check_file

    elif (
        prompt == "A video of two adult roe deer sniffing while reacting to the camera."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            cam_reaction_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION.name.lower()
            )
            adult_roe_deer_reaction = get_deer_tracks_from_age(
                cam_reaction_tracks,
                age=DAge.ADULT.name.lower(),
                deer_species=Species.ROE_DEER.name.lower(),
            )
            return (
                get_nb_tracks_action_in_video(
                    adult_roe_deer_reaction, action_name=Action.SNIFFING.name.lower()
                )
                == 2
            )

        return check_file

    elif prompt == "A video of a female adult red deer reacting to the camera.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            cam_reaction_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION.name.lower()
            )
            return tracks_contain_adult_deer_sex(
                cam_reaction_tracks,
                sex=DSex.FEMALE.name.lower(),
                deer_species=Species.RED_DEER.name.lower(),
            )

        return check_file

    elif (
        prompt == "A video of a male red deer running while participating in courtship."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            male_red_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.MALE.name.lower(),
                deer_species=Species.RED_DEER.name.lower(),
            )
            male_courtship_tracks = get_tracks_from_activity(
                male_red_deer_tracks, activity_name=Activity.COURTSHIP.name.lower()
            )
            return tracks_contain_action(
                male_courtship_tracks, action_name=Action.RUNNING.name.lower()
            )

        return check_file

    elif (
        prompt
        == "A video of a female red deer running while participating in courtship."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            female_red_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.FEMALE.name.lower(),
                deer_species=Species.RED_DEER.name.lower(),
            )
            female_courtship_tracks = get_tracks_from_activity(
                female_red_deer_tracks, activity_name=Activity.COURTSHIP.name.lower()
            )
            return tracks_contain_action(
                female_courtship_tracks, action_name=Action.RUNNING.name.lower()
            )

        return check_file

    elif (
        prompt
        == "A video of one male and one female red deer running while participating in courtship."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            male_red_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.MALE.name.lower(),
                deer_species=Species.RED_DEER.name.lower(),
            )
            male_courtship_tracks = get_tracks_from_activity(
                male_red_deer_tracks, activity_name=Activity.COURTSHIP.name.lower()
            )

            female_red_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.FEMALE.name.lower(),
                deer_species=Species.RED_DEER.name.lower(),
            )
            female_courtship_tracks = get_tracks_from_activity(
                female_red_deer_tracks, activity_name=Activity.COURTSHIP.name.lower()
            )

            return tracks_contain_action(
                male_courtship_tracks, action_name=Action.RUNNING.name.lower()
            ) and tracks_contain_action(
                female_courtship_tracks, action_name=Action.RUNNING.name.lower()
            )

        return check_file

    elif (
        prompt == "A video of a male roe deer running while participating in courtship."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            male_roe_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.MALE.name.lower(),
                deer_species=Species.ROE_DEER.name.lower(),
            )
            male_courtship_tracks = get_tracks_from_activity(
                male_roe_deer_tracks, activity_name=Activity.COURTSHIP.name.lower()
            )
            return tracks_contain_action(
                male_courtship_tracks, action_name=Action.RUNNING.name.lower()
            )

        return check_file

    elif (
        prompt
        == "A video of a female roe deer running while participating in courtship."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            female_roe_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.FEMALE.name.lower(),
                deer_species=Species.ROE_DEER.name.lower(),
            )
            female_courtship_tracks = get_tracks_from_activity(
                female_roe_deer_tracks, activity_name=Activity.COURTSHIP.name.lower()
            )
            return tracks_contain_action(
                female_courtship_tracks, action_name=Action.RUNNING.name.lower()
            )

        return check_file

    elif (
        prompt
        == "A video of one male and one female roe deer running while participating in courtship."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            male_roe_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.MALE.name.lower(),
                deer_species=Species.ROE_DEER.name.lower(),
            )
            male_courtship_tracks = get_tracks_from_activity(
                male_roe_deer_tracks, activity_name=Activity.COURTSHIP.name.lower()
            )

            female_roe_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.FEMALE.name.lower(),
                deer_species=Species.ROE_DEER.name.lower(),
            )
            female_courtship_tracks = get_tracks_from_activity(
                female_roe_deer_tracks, activity_name=Activity.COURTSHIP.name.lower()
            )

            return tracks_contain_action(
                male_courtship_tracks, action_name=Action.RUNNING.name.lower()
            ) and tracks_contain_action(
                female_courtship_tracks, action_name=Action.RUNNING.name.lower()
            )

        return check_file

    elif prompt == "A video of a fox sniffing while foraging.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            fox_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.FOX.name.lower()
            )
            fox_foraging_tracks = get_tracks_from_activity(
                fox_tracks, activity_name=Activity.FORAGING.name.lower()
            )

            return tracks_contain_action(
                fox_foraging_tracks, action_name=Action.SNIFFING.name.lower()
            )

        return check_file

    elif prompt == "A video of a fox hunting.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            fox_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.FOX.name.lower()
            )

            return tracks_contain_activity(
                fox_tracks, activity_name=Activity.HUNTING.name.lower()
            )

        return check_file

    elif prompt == "A video of a wolf hunting.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            wolf_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.WOLF.name.lower()
            )

            return tracks_contain_activity(
                wolf_tracks, activity_name=Activity.HUNTING.name.lower()
            )

        return check_file

    elif prompt == "A video of a hare.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            return tracks_contain_species(
                individual_tracks, species_name=Species.MOUNTAIN_HARE.name.lower()
            )

        return check_file

    elif prompt == "A video of a marten.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            return tracks_contain_species(
                individual_tracks, species_name=Species.MARTEN.name.lower()
            )

        return check_file

    elif prompt == "A video of a juvenile red deer playing.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            juvenile_red_deer = get_deer_tracks_from_age(
                individual_tracks,
                age=DAge.JUVENILE.name.lower(),
                deer_species=Species.RED_DEER.name.lower(),
            )

            return tracks_contain_activity(
                juvenile_red_deer, activity_name=Activity.PLAYING.name.lower()
            )

        return check_file

    elif prompt == "A video of a adult red deer browsing.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            adult_red_deer = get_deer_tracks_from_age(
                individual_tracks,
                age=DAge.ADULT.name.lower(),
                deer_species=Species.RED_DEER.name.lower(),
            )

            return tracks_contain_action(
                adult_red_deer, action_name=Action.BROWSING.name.lower()
            )

        return check_file

    elif prompt == "A video of an animal running while reacting to the camera.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            cam_reaction_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION.name.lower()
            )

            return tracks_contain_action(
                cam_reaction_tracks, action_name=Action.RUNNING.name.lower()
            )

        return check_file

    elif prompt == "A video of an animal looking at a camera.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            return tracks_contain_action(
                individual_tracks, action_name=Action.LOOKING_AT_CAMERA.name.lower()
            )

        return check_file

    else:
        raise NotImplementedError(
            f"No parsing function implemented yet for prompt '{prompt}'"
        )
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-IJ",
        "--json_folder",
        type=str,
        required=True,
        help="Path to the folder containing JSON annotation files.",
    )
    args = parser.parse_args()
    json_folder = Path(args.json_folder)

    with open("queries_and_videos.json", "r") as f:
        queries_dict = json.load(f)

    queries_list = [q for q_cat in queries_dict.values() for q in q_cat]

    for q_cat in queries_dict:
        for q in queries_dict[q_cat]:
            corresponding_files = [
                f.stem
                for f in json_folder.rglob("*/*.json")
                if get_parsing_function(q)(f)
            ]
            print(q, ":", len(corresponding_files), "videos")
            queries_dict[q_cat][q] = corresponding_files

    with open("queries_and_videos.json", "w") as f:
        json.dump(queries_dict, f, indent=2)
