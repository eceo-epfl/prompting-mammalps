import argparse
import json

from parse_json import *
from tqdm import tqdm


def get_tracks_from_json(json_file):

    video_detections = load_video_detections(json_file)
    individual_tracks = get_tracks_from_video_detections(video_detections)

    return individual_tracks


def get_parsing_function(prompt):
    if prompt == "An animal doing any other activity than foraging.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            activities_set = get_unique_activities_from_tracks(individual_tracks)
            return activities_set != set([Activity.FORAGING])

        return check_file

    elif prompt == "An animal that is neither a red deer nor a roe deer.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            species_set = get_unique_species_from_tracks(individual_tracks)
            return len(species_set - set([Species.RED_DEER, Species.ROE_DEER])) > 0

        return check_file

    elif prompt == "An animal running.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            return tracks_contain_action(
                individual_tracks, action_name=Action.TROTTING_OR_RUNNING
            )

        return check_file

    elif prompt == "An animal bathing.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            return tracks_contain_action(individual_tracks, action_name=Action.BATHING)

        return check_file

    elif prompt == "A roe deer grazing.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            roe_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.ROE_DEER
            )
            return tracks_contain_action(roe_deer_tracks, action_name=Action.GRAZING)

        return check_file

    elif prompt == "An animal browsing.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            return tracks_contain_action(individual_tracks, action_name=Action.BROWSING)

        return check_file

    elif prompt == "A female adult roe deer sniffing.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            female_roe_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.FEMALE,
                deer_species=Species.ROE_DEER,
            )
            return tracks_contain_action(
                female_roe_deer_tracks, action_name=Action.SNIFFING
            )

        return check_file

    elif prompt == "A juvenile red deer scratching its body.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            juvenile_red_deer_tracks = get_deer_tracks_from_age(
                individual_tracks,
                age=DAge.JUVENILE,
                deer_species=Species.RED_DEER,
            )
            return tracks_contain_action(
                juvenile_red_deer_tracks,
                action_name=Action.SCRATCHING_OWN_HEAD_OR_BODY,
            )

        return check_file

    elif prompt == "A juvenile roe deer preparing to suckle.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            juvenile_roe_deer_tracks = get_deer_tracks_from_age(
                individual_tracks,
                age=DAge.JUVENILE,
                deer_species=Species.ROE_DEER,
            )
            return tracks_contain_action(
                juvenile_roe_deer_tracks,
                action_name=Action.PREPARING_TO_SUCKLE,
            )

        return check_file

    elif prompt == "A chamois resting.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            chamois_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.CHAMOIS
            )
            return tracks_contain_activity(
                chamois_tracks, activity_name=Activity.RESTING
            )

        return check_file

    elif prompt == "An adult red deer reacting to a camera.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            adult_red_deer_tracks = get_deer_tracks_from_age(
                individual_tracks,
                age=DAge.ADULT,
                deer_species=Species.RED_DEER,
            )
            return tracks_contain_activity(
                adult_red_deer_tracks,
                activity_name=Activity.CAMERA_REACTION,
            )

        return check_file

    elif prompt == "A male adult roe deer jumping.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            male_roe_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.MALE,
                deer_species=Species.ROE_DEER,
            )
            return tracks_contain_action(
                male_roe_deer_tracks, action_name=Action.JUMPING
            )

        return check_file

    elif prompt == "A male adult red deer rubbing its antlers on the ground.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            male_red_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.MALE,
                deer_species=Species.RED_DEER,
            )
            return tracks_contain_action(
                male_red_deer_tracks,
                action_name=Action.RUBBING_ANTLERS_ON_GROUND,
            )

        return check_file

    elif (
        prompt == "An adult red deer standing head up while participating in courtship."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            individual_courtship = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.COURTSHIP
            )
            adult_deer_courtship = get_deer_tracks_from_age(
                individual_courtship,
                age=DAge.ADULT,
                deer_species=Species.RED_DEER,
            )
            return tracks_contain_action(
                adult_deer_courtship, action_name=Action.STANDING_HEAD_UP
            )

        return check_file

    elif prompt == "An adult red deer vocalizing while participating in courtship.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            individual_courtship = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.COURTSHIP
            )
            adult_deer_courtship = get_deer_tracks_from_age(
                individual_courtship,
                age=DAge.ADULT,
                deer_species=Species.RED_DEER,
            )
            return tracks_contain_action(
                adult_deer_courtship, action_name=Action.VOCALIZING
            )

        return check_file

    elif (
        prompt
        == "A video of two adult red deer vocalizing while participating in courtship."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            individual_courtship = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.COURTSHIP
            )
            adult_deer_courtship = get_deer_tracks_from_age(
                individual_courtship,
                age=DAge.ADULT,
                deer_species=Species.RED_DEER,
            )
            return (
                get_nb_tracks_action_in_video(
                    adult_deer_courtship, action_name=Action.VOCALIZING
                )
                == 2
            )

        return check_file

    elif prompt == "An adult red deer laying down while participating in courtship.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            individual_courtship = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.COURTSHIP
            )
            adult_deer_courtship = get_deer_tracks_from_age(
                individual_courtship,
                age=DAge.ADULT,
                deer_species=Species.RED_DEER,
            )
            return tracks_contain_action(
                adult_deer_courtship, action_name=Action.LAYING_DOWN
            )

        return check_file

    elif prompt == "A video of three or more red deer.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            return (
                get_nb_tracks_species_in_video(
                    individual_tracks, species_name=Species.RED_DEER
                )
                >= 3
            )

        return check_file

    elif prompt == "A video of two red deer.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            return (
                get_nb_tracks_species_in_video(
                    individual_tracks, species_name=Species.RED_DEER
                )
                == 2
            )

        return check_file

    elif prompt == "A video of three or more red deer reacting to a camera.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            red_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.RED_DEER
            )
            return (
                get_nb_tracks_activity_in_video(
                    red_deer_tracks, activity_name=Activity.CAMERA_REACTION
                )
                >= 3
            )

        return check_file

    elif prompt == "A video of three or more red deer escaping.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            red_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.RED_DEER
            )
            return (
                get_nb_tracks_activity_in_video(
                    red_deer_tracks, activity_name=Activity.ESCAPING
                )
                >= 3
            )

        return check_file

    elif prompt == "A juvenile red deer walking while in vigilance.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            juvenile_red_deer_tracks = get_deer_tracks_from_age(
                individual_tracks,
                age=DAge.JUVENILE,
                deer_species=Species.RED_DEER,
            )
            juvenile_red_deer_tracks_vigilance = get_tracks_from_activity(
                juvenile_red_deer_tracks, activity_name=Activity.VIGILANCE
            )
            return tracks_contain_action(
                juvenile_red_deer_tracks_vigilance,
                action_name=Action.WALKING,
            )

        return check_file

    elif prompt == "A video of two female adult red deer walking while foraging.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            female_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.FEMALE,
                deer_species=Species.RED_DEER,
            )
            female_deer_tracks_foraging = get_tracks_from_activity(
                female_deer_tracks, activity_name=Activity.FORAGING
            )
            return (
                get_nb_tracks_action_in_video(
                    female_deer_tracks_foraging, action_name=Action.WALKING
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
                sex=DSex.MALE,
                deer_species=Species.RED_DEER,
            )
            male_deer_tracks_foraging = get_tracks_from_activity(
                male_deer_tracks, activity_name=Activity.FORAGING
            )
            return (
                get_nb_tracks_action_in_video(
                    male_deer_tracks_foraging, action_name=Action.WALKING
                )
                >= 3
            )

        return check_file

    elif prompt == "An adult roe deer reacting to a camera.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            cam_reaction_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION
            )
            return tracks_contain_deer_age(
                cam_reaction_tracks,
                age=DAge.ADULT,
                deer_species=Species.ROE_DEER,
            )

        return check_file

    elif prompt == "A juvenile roe deer reacting to a camera.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            cam_reaction_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION
            )
            return tracks_contain_deer_age(
                cam_reaction_tracks,
                age=DAge.JUVENILE,
                deer_species=Species.ROE_DEER,
            )

        return check_file

    elif prompt == "A male adult red deer reacting to a camera.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            cam_reaction_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION
            )
            return tracks_contain_adult_deer_sex(
                cam_reaction_tracks,
                sex=DSex.MALE,
                deer_species=Species.RED_DEER,
            )

        return check_file

    elif prompt == "A juvenile red deer reacting to a camera.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            cam_reaction_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION
            )
            return tracks_contain_deer_age(
                cam_reaction_tracks,
                age=DAge.JUVENILE,
                deer_species=Species.RED_DEER,
            )

        return check_file

    elif prompt == "A female adult roe deer reacting to a camera.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            cam_reaction_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION
            )
            return tracks_contain_adult_deer_sex(
                cam_reaction_tracks,
                sex=DSex.FEMALE,
                deer_species=Species.ROE_DEER,
            )

        return check_file

    elif prompt == "A video of two adult roe deer sniffing while reacting to a camera.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            cam_reaction_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION
            )
            adult_roe_deer_reaction = get_deer_tracks_from_age(
                cam_reaction_tracks,
                age=DAge.ADULT,
                deer_species=Species.ROE_DEER,
            )
            return (
                get_nb_tracks_action_in_video(
                    adult_roe_deer_reaction, action_name=Action.SNIFFING
                )
                == 2
            )

        return check_file

    elif prompt == "A female adult red deer reacting to a camera.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            cam_reaction_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION
            )
            return tracks_contain_adult_deer_sex(
                cam_reaction_tracks,
                sex=DSex.FEMALE,
                deer_species=Species.RED_DEER,
            )

        return check_file

    elif prompt == "An adult male red deer running while participating in courtship.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            male_red_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.MALE,
                deer_species=Species.RED_DEER,
            )
            male_courtship_tracks = get_tracks_from_activity(
                male_red_deer_tracks, activity_name=Activity.COURTSHIP
            )
            return tracks_contain_action(
                male_courtship_tracks,
                action_name=Action.TROTTING_OR_RUNNING,
            )

        return check_file

    elif prompt == "An adult female red deer running while participating in courtship.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            female_red_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.FEMALE,
                deer_species=Species.RED_DEER,
            )
            female_courtship_tracks = get_tracks_from_activity(
                female_red_deer_tracks, activity_name=Activity.COURTSHIP
            )
            return tracks_contain_action(
                female_courtship_tracks,
                action_name=Action.TROTTING_OR_RUNNING,
            )

        return check_file

    elif (
        prompt
        == "A video of one adult male and one adult female red deer running while participating in courtship."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            male_red_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.MALE,
                deer_species=Species.RED_DEER,
            )
            male_courtship_tracks = get_tracks_from_activity(
                male_red_deer_tracks, activity_name=Activity.COURTSHIP
            )

            female_red_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.FEMALE,
                deer_species=Species.RED_DEER,
            )
            female_courtship_tracks = get_tracks_from_activity(
                female_red_deer_tracks, activity_name=Activity.COURTSHIP
            )

            return tracks_contain_action(
                male_courtship_tracks,
                action_name=Action.TROTTING_OR_RUNNING,
            ) and tracks_contain_action(
                female_courtship_tracks,
                action_name=Action.TROTTING_OR_RUNNING,
            )

        return check_file

    elif prompt == "An adult male roe deer running while participating in courtship.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            male_roe_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.MALE,
                deer_species=Species.ROE_DEER,
            )
            male_courtship_tracks = get_tracks_from_activity(
                male_roe_deer_tracks, activity_name=Activity.COURTSHIP
            )
            return tracks_contain_action(
                male_courtship_tracks,
                action_name=Action.TROTTING_OR_RUNNING,
            )

        return check_file

    elif prompt == "An adult female roe deer running while participating in courtship.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            female_roe_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.FEMALE,
                deer_species=Species.ROE_DEER,
            )
            female_courtship_tracks = get_tracks_from_activity(
                female_roe_deer_tracks, activity_name=Activity.COURTSHIP
            )
            return tracks_contain_action(
                female_courtship_tracks,
                action_name=Action.TROTTING_OR_RUNNING,
            )

        return check_file

    elif (
        prompt
        == "A video of one adult male and one adult female roe deer running while participating in courtship."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            male_roe_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.MALE,
                deer_species=Species.ROE_DEER,
            )
            male_courtship_tracks = get_tracks_from_activity(
                male_roe_deer_tracks, activity_name=Activity.COURTSHIP
            )

            female_roe_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.FEMALE,
                deer_species=Species.ROE_DEER,
            )
            female_courtship_tracks = get_tracks_from_activity(
                female_roe_deer_tracks, activity_name=Activity.COURTSHIP
            )

            return tracks_contain_action(
                male_courtship_tracks,
                action_name=Action.TROTTING_OR_RUNNING,
            ) and tracks_contain_action(
                female_courtship_tracks,
                action_name=Action.TROTTING_OR_RUNNING,
            )

        return check_file

    elif prompt == "A fox sniffing while foraging.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            fox_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.FOX
            )
            fox_foraging_tracks = get_tracks_from_activity(
                fox_tracks, activity_name=Activity.FORAGING
            )

            return tracks_contain_action(
                fox_foraging_tracks, action_name=Action.SNIFFING
            )

        return check_file

    elif prompt == "A fox chasing.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            fox_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.FOX
            )

            return tracks_contain_activity(fox_tracks, activity_name=Activity.CHASING)

        return check_file

    elif prompt == "A wolf chasing.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            wolf_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.WOLF
            )

            return tracks_contain_activity(wolf_tracks, activity_name=Activity.CHASING)

        return check_file

    elif prompt == "A hare.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            return tracks_contain_species(individual_tracks, species_name=Species.HARE)

        return check_file

    elif prompt == "A marten.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            return tracks_contain_species(
                individual_tracks, species_name=Species.MARTEN
            )

        return check_file

    elif prompt == "A juvenile red deer playing.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            juvenile_red_deer = get_deer_tracks_from_age(
                individual_tracks,
                age=DAge.JUVENILE,
                deer_species=Species.RED_DEER,
            )

            return tracks_contain_activity(
                juvenile_red_deer, activity_name=Activity.PLAYING
            )

        return check_file

    elif prompt == "An adult red deer browsing.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            adult_red_deer = get_deer_tracks_from_age(
                individual_tracks,
                age=DAge.ADULT,
                deer_species=Species.RED_DEER,
            )

            return tracks_contain_action(adult_red_deer, action_name=Action.BROWSING)

        return check_file

    elif prompt == "An animal running while reacting to a camera.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            cam_reaction_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION
            )

            return tracks_contain_action(
                cam_reaction_tracks, action_name=Action.TROTTING_OR_RUNNING
            )

        return check_file

    elif prompt == "An animal looking at a camera.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            return tracks_contain_action(
                individual_tracks, action_name=Action.LOOKING_AT_CAMERA
            )

        return check_file

    elif prompt == "A hare foraging.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            hare_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.HARE
            )
            return tracks_contain_activity(hare_tracks, activity_name=Activity.FORAGING)

        return check_file

    elif prompt == "A chamois.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            return tracks_contain_species(
                individual_tracks, species_name=Species.CHAMOIS
            )

        return check_file

    elif prompt == "A chamois in vigilance.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            chamois_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.CHAMOIS
            )
            return tracks_contain_activity(
                chamois_tracks, activity_name=Activity.VIGILANCE
            )

        return check_file

    elif prompt == "A fox foraging.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            fox_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.FOX
            )
            return tracks_contain_activity(fox_tracks, activity_name=Activity.FORAGING)

        return check_file

    elif prompt == "A wolf foraging.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            wolf_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.WOLF
            )
            return tracks_contain_activity(wolf_tracks, activity_name=Activity.FORAGING)

        return check_file

    elif prompt == "A juvenile red deer shaking its head or its body.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            juvenile_red_deer_tracks = get_deer_tracks_from_age(
                individual_tracks,
                age=DAge.JUVENILE,
                deer_species=Species.RED_DEER,
            )
            return tracks_contain_action(
                juvenile_red_deer_tracks,
                action_name=Action.SHAKING_HEAD_OR_BODY,
            )

        return check_file

    elif prompt == "A juvenile roe deer foraging.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            juvenile_roe_deer_tracks = get_deer_tracks_from_age(
                individual_tracks,
                age=DAge.JUVENILE,
                deer_species=Species.ROE_DEER,
            )
            return tracks_contain_activity(
                juvenile_roe_deer_tracks, activity_name=Activity.FORAGING
            )

        return check_file

    elif prompt == "A roe deer trotting while foraging.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            roe_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.ROE_DEER
            )
            roe_deer_trotting_tracks = get_tracks_from_action(
                roe_deer_tracks, action_name=Action.TROTTING_OR_RUNNING
            )
            return tracks_contain_activity(
                roe_deer_trotting_tracks, activity_name=Activity.FORAGING
            )

        return check_file

    elif prompt == "A mountain hare in vigilance.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            hare_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.HARE
            )
            return tracks_contain_activity(
                hare_tracks, activity_name=Activity.VIGILANCE
            )

        return check_file

    elif prompt == "A mountain hare grazing.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            mountain_hare_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.HARE
            )
            return tracks_contain_action(
                mountain_hare_tracks, action_name=Action.GRAZING
            )

        return check_file

    elif prompt == "An animal stretching its body.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            return tracks_contain_action(
                individual_tracks, action_name=Action.STRETCHING_BODY
            )

        return check_file

    elif prompt == "An animal bathing while grooming.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            bathing_tracks = get_tracks_from_action(
                individual_tracks, action_name=Action.BATHING
            )
            return tracks_contain_activity(
                bathing_tracks, activity_name=Activity.GROOMING
            )

        return check_file

    elif prompt == "An animal resting.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            return tracks_contain_activity(
                individual_tracks, activity_name=Activity.RESTING
            )

        return check_file

    elif prompt == "A red deer browsing.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            red_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.RED_DEER
            )
            return tracks_contain_action(red_deer_tracks, action_name=Action.BROWSING)

        return check_file

    elif prompt == "An animal running while foraging.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            running_tracks = get_tracks_from_action(
                individual_tracks, action_name=Action.TROTTING_OR_RUNNING
            )
            return tracks_contain_activity(
                running_tracks, activity_name=Activity.FORAGING
            )

        return check_file

    elif prompt == "A video of an animal drinking.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            return tracks_contain_action(individual_tracks, action_name=Action.DRINKING)

        return check_file

    elif prompt == "A video of an animal laying down while resting.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            laying_down_tracks = get_tracks_from_action(
                individual_tracks, action_name=Action.LAYING_DOWN
            )
            return tracks_contain_activity(
                laying_down_tracks, activity_name=Activity.RESTING
            )

        return check_file
    elif prompt == "A red deer resting while it is raining.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            red_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.RED_DEER
            )
            return tracks_contain_activity(
                red_deer_tracks, activity_name=Activity.RESTING
            ) and check_contains_weather_condition(
                json_file, weather_condition=Meteo.RAINY
            )

        return check_file

    elif prompt == "A rainy weather.":

        def check_file(json_file):

            return check_contains_weather_condition(
                json_file, weather_condition=Meteo.RAINY
            )

        return check_file

    elif prompt == "An animal participating in courtship.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            return tracks_contain_activity(
                individual_tracks, activity_name=Activity.COURTSHIP
            )

        return check_file

    elif prompt == "An adult male red deer pawing the ground while wallowing.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            male_red_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.MALE,
                deer_species=Species.RED_DEER,
            )
            male_pawing_ground = get_tracks_from_action(
                male_red_deer_tracks, action_name=Action.PAWING_GROUND
            )

            return tracks_contain_activity(
                male_pawing_ground,
                activity_name=Activity.MARKING_OR_WALLOWING,
            )

        return check_file

    elif (
        prompt
        == "An adult male red deer pawing the ground and rubbing its antlers on the ground while wallowing."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            male_red_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.MALE,
                deer_species=Species.RED_DEER,
            )
            male_wallowing = get_tracks_from_activity(
                male_red_deer_tracks,
                activity_name=Activity.MARKING_OR_WALLOWING,
            )

            return tracks_contain_action(
                male_wallowing, action_name=Action.PAWING_GROUND
            ) and tracks_contain_action(
                male_wallowing,
                action_name=Action.RUBBING_ANTLERS_ON_GROUND,
            )

        return check_file

    elif prompt == "An adult male red deer bathing while wallowing.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            male_red_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.MALE,
                deer_species=Species.RED_DEER,
            )
            male_bathing = get_tracks_from_action(
                male_red_deer_tracks, action_name=Action.BATHING
            )

            return tracks_contain_activity(
                male_bathing, activity_name=Activity.MARKING_OR_WALLOWING
            )

        return check_file

    elif prompt == "An adult male red deer urinating while wallowing.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            male_red_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.MALE,
                deer_species=Species.RED_DEER,
            )
            male_urinating = get_tracks_from_action(
                male_red_deer_tracks, action_name=Action.URINATING
            )

            return tracks_contain_activity(
                male_urinating, activity_name=Activity.MARKING_OR_WALLOWING
            )

        return check_file

    elif prompt == "An adult male red deer in vigilance after vocalizing.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            male_red_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.MALE,
                deer_species=Species.RED_DEER,
            )
            sequence = [Action.VOCALIZING, Activity.VIGILANCE]
            for track in male_red_deer_tracks:
                if check_track_contains_continuous_sequence(track, sequence):
                    return True

            return False

        return check_file
    elif prompt == "A red deer vocalizing while it is raining.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            red_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.RED_DEER
            )
            vocalizing_tracks = get_tracks_from_action(
                red_deer_tracks, action_name=Action.VOCALIZING
            )
            return tracks_contain_action(
                vocalizing_tracks, action_name=Action.VOCALIZING
            ) and check_contains_weather_condition(
                json_file, weather_condition=Meteo.RAINY
            )

        return check_file
    elif (
        prompt
        == "A roe deer running in a rainy weather while participating in courtship."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            roe_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.ROE_DEER
            )
            courtship_tracks = get_tracks_from_activity(
                roe_deer_tracks, activity_name=Activity.COURTSHIP
            )
            running_courtship_tracks = get_tracks_from_action(
                courtship_tracks, action_name=Action.TROTTING_OR_RUNNING
            )
            return tracks_contain_action(
                running_courtship_tracks, action_name=Action.TROTTING_OR_RUNNING
            ) and check_contains_weather_condition(
                json_file, weather_condition=Meteo.RAINY
            )

        return check_file
    elif prompt == "A roe deer participating in courtship while it is raining.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            roe_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.ROE_DEER
            )
            courtship_tracks = get_tracks_from_activity(
                roe_deer_tracks, activity_name=Activity.COURTSHIP
            )
            return tracks_contain_activity(
                courtship_tracks, activity_name=Activity.COURTSHIP
            ) and check_contains_weather_condition(
                json_file, weather_condition=Meteo.RAINY
            )

        return check_file
    elif prompt == "A red deer participating in courtship while it is raining.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            red_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.RED_DEER
            )
            courtship_tracks = get_tracks_from_activity(
                red_deer_tracks, activity_name=Activity.COURTSHIP
            )
            return tracks_contain_activity(
                courtship_tracks, activity_name=Activity.COURTSHIP
            ) and check_contains_weather_condition(
                json_file, weather_condition=Meteo.RAINY
            )

        return check_file

    elif prompt == "A video of two or more animals.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            return len(individual_tracks) >= 2

        return check_file

    elif prompt == "A video of two or more wolves.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            return (
                get_nb_tracks_species_in_video(
                    individual_tracks, species_name=Species.WOLF
                )
                >= 2
            )

        return check_file

    elif (
        prompt == "An adult female red deer foraging and a juvenile red deer foraging."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            female_red_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.FEMALE,
                deer_species=Species.RED_DEER,
            )
            juvenile_red_deer_tracks = get_deer_tracks_from_age(
                individual_tracks,
                age=DAge.JUVENILE,
                deer_species=Species.RED_DEER,
            )

            return tracks_contain_activity(
                female_red_deer_tracks, activity_name=Activity.FORAGING
            ) and tracks_contain_activity(
                juvenile_red_deer_tracks, activity_name=Activity.FORAGING
            )

        return check_file

    elif prompt == "A video showing individuals from at least two different species.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            species_set = get_unique_species_from_tracks(individual_tracks)

            return len(species_set) >= 2

        return check_file

    elif prompt == "A juvenile red deer nursing.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            juvenile_red_deer_tracks = get_deer_tracks_from_age(
                individual_tracks,
                age=DAge.JUVENILE,
                deer_species=Species.RED_DEER,
            )

            return tracks_contain_activity(
                juvenile_red_deer_tracks, activity_name=Activity.NURSING
            )

        return check_file

    elif prompt == "An adult female red deer nursing.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            female_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.FEMALE,
                deer_species=Species.RED_DEER,
            )

            return tracks_contain_activity(
                female_deer_tracks, activity_name=Activity.NURSING
            )

        return check_file
    elif prompt == "A wolf chasing in a clear weather.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            wolf_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.WOLF
            )
            return tracks_contain_activity(
                wolf_tracks, activity_name=Activity.CHASING
            ) and check_contains_weather_condition(
                json_file, weather_condition=Meteo.CLEAR
            )

        return check_file
    elif prompt == "A wolf chasing in an overcast weather.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            wolf_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.WOLF
            )
            return tracks_contain_activity(
                wolf_tracks, activity_name=Activity.CHASING
            ) and check_contains_weather_condition(
                json_file, weather_condition=Meteo.OVERCAST
            )

        return check_file
    elif prompt == "A fox chasing in a clear weather.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            fox_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.FOX
            )
            return tracks_contain_activity(
                fox_tracks, activity_name=Activity.CHASING
            ) and check_contains_weather_condition(
                json_file, weather_condition=Meteo.CLEAR
            )

        return check_file
    elif prompt == "A female adult roe deer running in a rainy weather.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            female_roe_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.FEMALE,
                deer_species=Species.ROE_DEER,
            )
            return tracks_contain_activity(
                female_roe_deer_tracks, activity_name=Action.TROTTING_OR_RUNNING
            ) and check_contains_weather_condition(
                json_file, weather_condition=Meteo.RAINY
            )

        return check_file
    elif prompt == "A roe deer foraging in a sunny weather.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            roe_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.ROE_DEER
            )
            return tracks_contain_activity(
                roe_deer_tracks, activity_name=Activity.FORAGING
            ) and check_contains_weather_condition(
                json_file, weather_condition=Meteo.SUNNY
            )

        return check_file
    elif prompt == "A red deer foraging in a sunny weather.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            red_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.RED_DEER
            )
            return tracks_contain_activity(
                red_deer_tracks, activity_name=Activity.FORAGING
            ) and check_contains_weather_condition(
                json_file, weather_condition=Meteo.SUNNY
            )

        return check_file
    elif prompt == "A roe deer foraging in a rainy weather.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            roe_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.ROE_DEER
            )
            return tracks_contain_activity(
                roe_deer_tracks, activity_name=Activity.FORAGING
            ) and check_contains_weather_condition(
                json_file, weather_condition=Meteo.RAINY
            )

        return check_file
    elif prompt == "A red deer foraging in a rainy weather.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            red_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.RED_DEER
            )
            return tracks_contain_activity(
                red_deer_tracks, activity_name=Activity.FORAGING
            ) and check_contains_weather_condition(
                json_file, weather_condition=Meteo.RAINY
            )

        return check_file
    elif prompt == "A juvenile red deer playing in a clear weather.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            juvenile_red_deer_tracks = get_deer_tracks_from_age(
                individual_tracks,
                age=DAge.JUVENILE,
                deer_species=Species.RED_DEER,
            )
            return tracks_contain_activity(
                juvenile_red_deer_tracks, activity_name=Activity.PLAYING
            ) and check_contains_weather_condition(
                json_file, weather_condition=Meteo.CLEAR
            )

        return check_file
    elif prompt == "A female adult red deer escaping in a sunny weather.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            female_red_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks, sex=DSex.FEMALE, deer_species=Species.RED_DEER
            )
            return tracks_contain_activity(
                female_red_deer_tracks, activity_name=Activity.ESCAPING
            ) and check_contains_weather_condition(
                json_file, weather_condition=Meteo.SUNNY
            )

        return check_file
    elif prompt == "An animal in vigilance while the weather is rainy or overcast.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            return (
                tracks_contain_activity(
                    individual_tracks, activity_name=Activity.VIGILANCE
                )
                and check_contains_weather_condition(
                    json_file, weather_condition=Meteo.RAINY
                )
                or check_contains_weather_condition(
                    json_file, weather_condition=Meteo.OVERCAST
                )
            )

        return check_file
    elif prompt == "An animal in vigilance while the weather is clear or sunny.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            return (
                tracks_contain_activity(
                    individual_tracks, activity_name=Activity.VIGILANCE
                )
                and check_contains_weather_condition(
                    json_file, weather_condition=Meteo.CLEAR
                )
                or check_contains_weather_condition(
                    json_file, weather_condition=Meteo.SUNNY
                )
            )

        return check_file
    elif prompt == "A juvenile red deer suckling.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            juvenile_red_deer_tracks = get_deer_tracks_from_age(
                individual_tracks,
                age=DAge.JUVENILE,
                deer_species=Species.RED_DEER,
            )

            return tracks_contain_action(
                juvenile_red_deer_tracks, action_name=Action.SUCKLING
            )

        return check_file

    elif prompt == "A juvenile roe deer suckling.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            juvenile_roe_deer_tracks = get_deer_tracks_from_age(
                individual_tracks,
                age=DAge.JUVENILE,
                deer_species=Species.ROE_DEER,
            )

            return tracks_contain_action(
                juvenile_roe_deer_tracks, action_name=Action.SUCKLING
            )

        return check_file

    elif prompt == "A video of two or more juvenile red deer.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            return (
                get_nb_deer_tracks_age_in_video(
                    individual_tracks,
                    age=DAge.JUVENILE,
                    deer_species=Species.RED_DEER,
                )
                >= 2
            )

        return check_file

    elif prompt == "An animal reacting to a camera.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            return tracks_contain_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION
            )

        return check_file

    elif prompt == "A fox reacting to a camera.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            fox_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.FOX
            )

            return tracks_contain_activity(
                fox_tracks, activity_name=Activity.CAMERA_REACTION
            )

        return check_file

    elif prompt == "A wolf reacting to a camera.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            wolf_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.WOLF
            )

            return tracks_contain_activity(
                wolf_tracks, activity_name=Activity.CAMERA_REACTION
            )

        return check_file

    elif prompt == "An animal reacting to a camera and then foraging.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            sequence = [Activity.CAMERA_REACTION, Activity.FORAGING]
            for track in individual_tracks:
                if check_track_contains_continuous_sequence(track, sequence):
                    return True

            return False

        return check_file

    elif (
        prompt
        == "An animal foraging, then reacting to a camera and then going back to foraging."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            sequence = [Activity.FORAGING, Activity.CAMERA_REACTION, Activity.FORAGING]
            for track in individual_tracks:
                if check_track_contains_continuous_sequence(track, sequence):
                    return True

            return False

        return check_file

    elif prompt == "An animal reacting to a camera and then running away.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            sequence = [Activity.CAMERA_REACTION, Action.TROTTING_OR_RUNNING]
            for track in individual_tracks:
                if check_track_contains_continuous_sequence(track, sequence):
                    return True

            return False

        return check_file

    elif (
        prompt == "An animal foraging, then reacting to a camera and then running away."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            sequence = [
                Activity.FORAGING,
                Activity.CAMERA_REACTION,
                Action.TROTTING_OR_RUNNING,
            ]

            for track in individual_tracks:
                if check_track_contains_continuous_sequence(track, sequence):
                    return True

            return False

        return check_file
    elif (
        prompt
        == "An animal reacting to the camera while the weather is rainy or overcast."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            return (
                tracks_contain_activity(
                    individual_tracks, activity_name=Activity.CAMERA_REACTION
                )
                and check_contains_weather_condition(
                    json_file, weather_condition=Meteo.RAINY
                )
                or check_contains_weather_condition(
                    json_file, weather_condition=Meteo.OVERCAST
                )
            )

        return check_file
    elif (
        prompt
        == "An animal reacting to the camera while the weather is clear or sunny."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            return (
                tracks_contain_activity(
                    individual_tracks, activity_name=Activity.CAMERA_REACTION
                )
                and check_contains_weather_condition(
                    json_file, weather_condition=Meteo.CLEAR
                )
                or check_contains_weather_condition(
                    json_file, weather_condition=Meteo.SUNNY
                )
            )

        return check_file
    elif (
        prompt
        == "An animal escaping from another animal of the same species chasing it."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            escaping_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.ESCAPING
            )
            species_escaping = get_unique_species_from_tracks(escaping_tracks)

            chasing_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CHASING
            )
            species_chasing = get_unique_species_from_tracks(chasing_tracks)

            return len(species_escaping.intersection(species_chasing)) > 0

        return check_file

    elif (
        prompt
        == "An animal escaping from another animal of a different species chasing it."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            escaping_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.ESCAPING
            )
            species_escaping = get_unique_species_from_tracks(escaping_tracks)

            chasing_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CHASING
            )
            species_chasing = get_unique_species_from_tracks(chasing_tracks)

            return (
                len(species_escaping) > 0
                and len(species_chasing) > 0
                and len(species_escaping.union(species_chasing)) > len(species_escaping)
            )

        return check_file

    elif (
        prompt
        == "Two or more red deer foraging and at least one is also in vigilance at some point."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            red_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.RED_DEER
            )

            return (
                get_nb_tracks_activity_in_video(
                    red_deer_tracks, activity_name=Activity.FORAGING
                )
                >= 2
            ) and (
                get_nb_tracks_activity_in_video(
                    red_deer_tracks, activity_name=Activity.VIGILANCE
                )
                >= 1
            )

        return check_file

    elif (
        prompt
        == "A juvenile red deer doing the exact same activities as an adult female red deer."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            juvenile_red_deer_tracks = get_deer_tracks_from_age(
                individual_tracks, age=DAge.JUVENILE, deer_species=Species.RED_DEER
            )
            female_red_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks, sex=DSex.FEMALE, deer_species=Species.RED_DEER
            )
            for j_track in juvenile_red_deer_tracks:
                j_track_activities = get_unique_activities_from_tracks([j_track])
                for a_track in female_red_deer_tracks:
                    a_track_activities = get_unique_activities_from_tracks([a_track])
                    if j_track_activities == a_track_activities:
                        return True

            return False

        return check_file

    elif (
        prompt
        == "A juvenile red deer doing the exact same actions as an adult female red deer."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            juvenile_red_deer_tracks = get_deer_tracks_from_age(
                individual_tracks, age=DAge.JUVENILE, deer_species=Species.RED_DEER
            )
            female_red_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks, sex=DSex.FEMALE, deer_species=Species.RED_DEER
            )
            for j_track in juvenile_red_deer_tracks:
                j_track_actions = get_unique_actions_from_tracks([j_track])
                for a_track in female_red_deer_tracks:
                    a_track_actions = get_unique_actions_from_tracks([a_track])
                    if j_track_actions == a_track_actions:
                        return True

            return False

        return check_file

    elif (
        prompt
        == "A juvenile red deer doing at least one different action than an adult female red deer."
    ):

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)

            juvenile_red_deer_tracks = get_deer_tracks_from_age(
                individual_tracks, age=DAge.JUVENILE, deer_species=Species.RED_DEER
            )
            female_red_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks, sex=DSex.FEMALE, deer_species=Species.RED_DEER
            )
            for j_track in juvenile_red_deer_tracks:
                j_track_actions = get_unique_actions_from_tracks([j_track])
                for a_track in female_red_deer_tracks:
                    a_track_actions = get_unique_actions_from_tracks([a_track])
                    if j_track_actions != a_track_actions:
                        return True

            return False

        return check_file

    elif prompt == "A single adult red deer only foraging.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            if len(individual_tracks) != 1:
                return False
            else:
                adult_deer_tracks = get_deer_tracks_from_age(
                    individual_tracks,
                    age=DAge.ADULT,
                    deer_species=Species.RED_DEER,
                )
                activities = get_unique_activities_from_tracks(adult_deer_tracks)
                return ("foraging" in activities) and (len(activities) == 1)

        return check_file

    elif prompt == "An empty video.":

        def check_file(json_file):
            individual_tracks = get_tracks_from_json(json_file)
            return len(individual_tracks) == 0

        return check_file

    else:
        print(f"No parsing function implemented yet for prompt '{prompt}'")
        return lambda x: False


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

    with open("benchmark/queries_and_videos_empty.json", "r") as f:
        queries_dict = json.load(f)

    # Initialize output dict with empty lists for each query
    out_queries_dict = {
        q_cat: {q: [] for q in queries_dict[q_cat]} for q_cat in queries_dict
    }

    # Precompute parsing functions for all queries
    parsing_functions = {}
    for q_cat in queries_dict:
        for q in queries_dict[q_cat]:
            parsing_functions[q] = get_parsing_function(q)

    # Iterate over all files once, check all queries for each file
    for f in tqdm(json_folder.rglob("*/*.json")):
        for q_cat in queries_dict:
            for q in queries_dict[q_cat]:
                try:
                    if parsing_functions[q](f):
                        out_queries_dict[q_cat][q].append(f.stem)
                except Exception as e:
                    print(f"Could not parse {f} for {q}")
                    print(e)
                    continue

    # Print and save results
    for q_cat in queries_dict:
        for q in queries_dict[q_cat]:
            print(q, ":", len(out_queries_dict[q_cat][q]), "videos")

    with open("benchmark/queries_and_videos.json", "w") as f:
        json.dump(out_queries_dict, f, indent=2)
