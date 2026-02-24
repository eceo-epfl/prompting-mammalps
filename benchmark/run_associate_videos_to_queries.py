import argparse
import json

from parse_json import *
from tqdm import tqdm


def get_parsing_function(prompt):
    if prompt == "An animal engaged in any activity other than foraging.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            activities_set = get_unique_activities_from_tracks(individual_tracks)
            return activities_set != set([Activity.FORAGING])

        return check_file

    elif prompt == "An animal that is neither a red deer nor a roe deer.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            species_set = get_unique_species_from_tracks(individual_tracks)
            return len(species_set - set([Species.RED_DEER, Species.ROE_DEER])) > 0

        return check_file

    elif prompt == "An animal running.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            return tracks_contain_action(
                individual_tracks, action_name=Action.TROTTING_OR_RUNNING
            )

        return check_file

    elif prompt == "An animal bathing.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            return tracks_contain_action(individual_tracks, action_name=Action.BATHING)

        return check_file

    elif prompt == "A roe deer grazing.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            roe_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.ROE_DEER
            )
            return tracks_contain_action(roe_deer_tracks, action_name=Action.GRAZING)

        return check_file

    elif prompt == "An animal browsing.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            return tracks_contain_action(individual_tracks, action_name=Action.BROWSING)

        return check_file

    elif prompt == "An adult roe deer sniffing.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            adult_roe_deer_tracks = get_deer_tracks_from_age(
                individual_tracks,
                age=DAge.ADULT,
                deer_species=Species.ROE_DEER,
            )
            return tracks_contain_action(
                adult_roe_deer_tracks, action_name=Action.SNIFFING
            )

        return check_file

    elif prompt == "A juvenile red deer scratching its body.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
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

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
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

    elif prompt == "A chamois trotting.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            chamois_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.CHAMOIS
            )
            return tracks_contain_action(
                chamois_tracks, action_name=Action.TROTTING_OR_RUNNING
            )

        return check_file

    elif prompt == "An adult red deer reacting to a camera.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
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

    elif prompt == "An adult male roe deer jumping.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            male_roe_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.MALE,
                deer_species=Species.ROE_DEER,
            )
            return tracks_contain_action(
                male_roe_deer_tracks, action_name=Action.JUMPING
            )

        return check_file

    elif prompt == "An adult male red deer rubbing its antlers on the ground.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
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
        prompt
        == "An adult red deer standing with its head up while participating in courtship."
    ):

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
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

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
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

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
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

    elif prompt == "An adult red deer lying down while participating in courtship.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
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

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            return (
                get_nb_tracks_species_in_video(
                    individual_tracks, species_name=Species.RED_DEER
                )
                >= 3
            )

        return check_file

    elif prompt == "A video of two red deer.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            return (
                get_nb_tracks_species_in_video(
                    individual_tracks, species_name=Species.RED_DEER
                )
                == 2
            )

        return check_file

    elif prompt == "A video of three or more red deer reacting to a camera.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
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

    elif prompt == "A video of two or more red deer escaping.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            red_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.RED_DEER
            )
            return (
                get_nb_tracks_activity_in_video(
                    red_deer_tracks, activity_name=Activity.ESCAPING
                )
                >= 1
            )

        return check_file

    elif prompt == "A juvenile red deer walking while being vigilant.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
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

    elif prompt == "A video of two adult female red deer walking while foraging.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
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
        prompt == "A video of three or more adult male red deer walking while foraging."
    ):

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
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

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            cam_reaction_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION
            )
            return tracks_contain_deer_age(
                cam_reaction_tracks,
                age=DAge.ADULT,
                deer_species=Species.ROE_DEER,
            )

        return check_file

    elif prompt == "Two or more animals reacting to a camera.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            return (
                get_nb_tracks_activity_in_video(
                    individual_tracks, activity_name=Activity.CAMERA_REACTION
                )
                >= 2
            )

        return check_file

    elif prompt == "A juvenile deer reacting to a camera.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            cam_reaction_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION
            )
            return tracks_contain_deer_age(
                cam_reaction_tracks,
                age=DAge.JUVENILE,
            )

        return check_file

    elif prompt == "An adult male red deer reacting to a camera.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            cam_reaction_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION
            )
            return tracks_contain_adult_deer_sex(
                cam_reaction_tracks,
                sex=DSex.MALE,
                deer_species=Species.RED_DEER,
            )

        return check_file

    elif prompt == "A video of an individual sniffing while reacting to a camera.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            cam_reaction_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION
            )

            return tracks_contain_action(
                cam_reaction_tracks, action_name=Action.SNIFFING
            )

        return check_file

    elif prompt == "An adult female red deer reacting to a camera.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
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

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
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

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
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

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
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

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
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

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
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

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
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

    elif prompt == "A fox sniffing.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            fox_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.FOX
            )

            return tracks_contain_action(fox_tracks, action_name=Action.SNIFFING)

        return check_file

    elif prompt == "A fox chasing prey.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            fox_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.FOX
            )

            return tracks_contain_activity(fox_tracks, activity_name=Activity.CHASING)

        return check_file

    elif prompt == "A wolf chasing prey.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            wolf_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.WOLF
            )

            return tracks_contain_activity(wolf_tracks, activity_name=Activity.CHASING)

        return check_file

    elif prompt == "A hare.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            return tracks_contain_species(individual_tracks, species_name=Species.HARE)

        return check_file

    elif prompt == "A marten.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            return tracks_contain_species(
                individual_tracks, species_name=Species.MARTEN
            )

        return check_file

    elif prompt == "A juvenile red deer playing.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
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

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            adult_red_deer = get_deer_tracks_from_age(
                individual_tracks,
                age=DAge.ADULT,
                deer_species=Species.RED_DEER,
            )

            return tracks_contain_action(adult_red_deer, action_name=Action.BROWSING)

        return check_file

    elif prompt == "An animal running while reacting to a camera.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            cam_reaction_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION
            )

            return tracks_contain_action(
                cam_reaction_tracks, action_name=Action.TROTTING_OR_RUNNING
            )

        return check_file

    elif prompt == "An animal looking at a camera.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            return tracks_contain_action(
                individual_tracks, action_name=Action.LOOKING_AT_CAMERA
            )

        return check_file

    elif prompt == "A hare foraging.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            hare_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.HARE
            )
            return tracks_contain_activity(hare_tracks, activity_name=Activity.FORAGING)

        return check_file

    elif prompt == "A chamois.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            return tracks_contain_species(
                individual_tracks, species_name=Species.CHAMOIS
            )

        return check_file

    elif prompt == "A chamois being vigilant.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            chamois_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.CHAMOIS
            )
            return tracks_contain_activity(
                chamois_tracks, activity_name=Activity.VIGILANCE
            )

        return check_file

    elif prompt == "A fox foraging.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            fox_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.FOX
            )
            return tracks_contain_activity(fox_tracks, activity_name=Activity.FORAGING)

        return check_file

    elif prompt == "A wolf foraging.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            wolf_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.WOLF
            )
            return tracks_contain_activity(wolf_tracks, activity_name=Activity.FORAGING)

        return check_file

    elif prompt == "A juvenile red deer shaking its head or body.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

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

    elif prompt == "A juvenile roe deer.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            juvenile_roe_deer_tracks = get_deer_tracks_from_age(
                individual_tracks,
                age=DAge.JUVENILE,
                deer_species=Species.ROE_DEER,
            )
            return len(juvenile_roe_deer_tracks) > 0

        return check_file

    elif prompt == "A roe deer trotting while foraging.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

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

    elif prompt == "A mountain hare being vigilant.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            hare_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.HARE
            )
            return tracks_contain_activity(
                hare_tracks, activity_name=Activity.VIGILANCE
            )

        return check_file

    elif prompt == "A mountain hare grazing.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            mountain_hare_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.HARE
            )
            return tracks_contain_action(
                mountain_hare_tracks, action_name=Action.GRAZING
            )

        return check_file

    elif prompt == "An animal stretching its body.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            return tracks_contain_action(
                individual_tracks, action_name=Action.STRETCHING_BODY
            )

        return check_file

    elif prompt == "An animal bathing while grooming.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            bathing_tracks = get_tracks_from_action(
                individual_tracks, action_name=Action.BATHING
            )
            return tracks_contain_activity(
                bathing_tracks, activity_name=Activity.GROOMING
            )

        return check_file

    elif prompt == "An animal resting.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            return tracks_contain_activity(
                individual_tracks, activity_name=Activity.RESTING
            )

        return check_file

    elif prompt == "A red deer browsing.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            red_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.RED_DEER
            )
            return tracks_contain_action(red_deer_tracks, action_name=Action.BROWSING)

        return check_file

    elif prompt == "An animal running while foraging.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            running_tracks = get_tracks_from_action(
                individual_tracks, action_name=Action.TROTTING_OR_RUNNING
            )
            return tracks_contain_activity(
                running_tracks, activity_name=Activity.FORAGING
            )

        return check_file

    elif prompt == "A video of an animal drinking.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            return tracks_contain_action(individual_tracks, action_name=Action.DRINKING)

        return check_file

    elif prompt == "A video of an animal lying down while resting.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            laying_down_tracks = get_tracks_from_action(
                individual_tracks, action_name=Action.LAYING_DOWN
            )
            return tracks_contain_activity(
                laying_down_tracks, activity_name=Activity.RESTING
            )

        return check_file
    elif prompt == "A red deer resting in rainy weather.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            red_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.RED_DEER
            )
            return tracks_contain_activity(
                red_deer_tracks, activity_name=Activity.RESTING
            ) and check_contains_weather_condition(
                video_id, weather_condition=Meteo.RAINY
            )

        return check_file

    elif prompt == "A red deer resting in clear weather.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            red_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.RED_DEER
            )
            return tracks_contain_activity(
                red_deer_tracks, activity_name=Activity.RESTING
            ) and check_contains_weather_condition(
                video_id, weather_condition=Meteo.CLEAR
            )

        return check_file

    elif prompt == "Rainy weather.":

        def check_file(video_id):

            return check_contains_weather_condition(
                video_id, weather_condition=Meteo.RAINY
            )

        return check_file

    elif prompt == "An animal participating in courtship.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            return tracks_contain_activity(
                individual_tracks, activity_name=Activity.COURTSHIP
            )

        return check_file

    elif prompt == "An adult male red deer pawing the ground while wallowing.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

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

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

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

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

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

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

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

    elif prompt == "An adult male red deer being vigilant after vocalizing.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

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
    elif prompt == "A red deer vocalizing in rainy weather.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            red_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.RED_DEER
            )
            vocalizing_tracks = get_tracks_from_action(
                red_deer_tracks, action_name=Action.VOCALIZING
            )
            return tracks_contain_action(
                vocalizing_tracks, action_name=Action.VOCALIZING
            ) and check_contains_weather_condition(
                video_id, weather_condition=Meteo.RAINY
            )

        return check_file

    elif prompt == "A roe deer participating in courtship in rainy weather.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            roe_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.ROE_DEER
            )
            return tracks_contain_activity(
                roe_deer_tracks, activity_name=Activity.COURTSHIP
            ) and check_contains_weather_condition(
                video_id, weather_condition=Meteo.RAINY
            )

        return check_file
    elif prompt == "A roe deer participating in courtship in clear weather.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            roe_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.ROE_DEER
            )

            return tracks_contain_activity(
                roe_deer_tracks, activity_name=Activity.COURTSHIP
            ) and check_contains_weather_condition(
                video_id, weather_condition=Meteo.CLEAR
            )

        return check_file
    elif prompt == "A red deer participating in courtship in rainy weather.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            red_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.RED_DEER
            )
            courtship_tracks = get_tracks_from_activity(
                red_deer_tracks, activity_name=Activity.COURTSHIP
            )
            return tracks_contain_activity(
                courtship_tracks, activity_name=Activity.COURTSHIP
            ) and check_contains_weather_condition(
                video_id, weather_condition=Meteo.RAINY
            )

        return check_file

    elif prompt == "A video of two or more animals.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            return len(individual_tracks) >= 2

        return check_file

    elif prompt == "A video of two or more wolves.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

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

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

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

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            species_set = get_unique_species_from_tracks(individual_tracks)

            return len(species_set) >= 2

        return check_file

    elif prompt == "A juvenile red deer nursing.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

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

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            female_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.FEMALE,
                deer_species=Species.RED_DEER,
            )

            return tracks_contain_activity(
                female_deer_tracks, activity_name=Activity.NURSING
            )

        return check_file
    elif prompt == "A wolf chasing prey in clear or sunny weather.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            wolf_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.WOLF
            )
            return tracks_contain_activity(
                wolf_tracks, activity_name=Activity.CHASING
            ) and (
                check_contains_weather_condition(
                    video_id, weather_condition=Meteo.CLEAR
                )
                or check_contains_weather_condition(
                    video_id, weather_condition=Meteo.SUNNY
                )
            )

        return check_file
    elif prompt == "A wolf chasing prey in overcast or rainy weather.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            wolf_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.WOLF
            )
            return tracks_contain_activity(
                wolf_tracks, activity_name=Activity.CHASING
            ) and (
                check_contains_weather_condition(
                    video_id, weather_condition=Meteo.OVERCAST
                )
                or check_contains_weather_condition(
                    video_id, weather_condition=Meteo.RAINY
                )
            )

        return check_file
    elif prompt == "A fox chasing prey in clear weather.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            fox_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.FOX
            )
            return tracks_contain_activity(
                fox_tracks, activity_name=Activity.CHASING
            ) and check_contains_weather_condition(
                video_id, weather_condition=Meteo.CLEAR
            )

        return check_file
    elif prompt == "An adult female roe deer running in rainy weather.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            female_roe_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.FEMALE,
                deer_species=Species.ROE_DEER,
            )
            return tracks_contain_activity(
                female_roe_deer_tracks, activity_name=Action.TROTTING_OR_RUNNING
            ) and check_contains_weather_condition(
                video_id, weather_condition=Meteo.RAINY
            )

        return check_file
    elif prompt == "A roe deer foraging in sunny weather.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            roe_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.ROE_DEER
            )
            return tracks_contain_activity(
                roe_deer_tracks, activity_name=Activity.FORAGING
            ) and check_contains_weather_condition(
                video_id, weather_condition=Meteo.SUNNY
            )

        return check_file
    elif prompt == "A red deer foraging in sunny weather.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            red_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.RED_DEER
            )
            return tracks_contain_activity(
                red_deer_tracks, activity_name=Activity.FORAGING
            ) and check_contains_weather_condition(
                video_id, weather_condition=Meteo.SUNNY
            )

        return check_file
    elif prompt == "A roe deer foraging in rainy weather.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            roe_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.ROE_DEER
            )
            return tracks_contain_activity(
                roe_deer_tracks, activity_name=Activity.FORAGING
            ) and check_contains_weather_condition(
                video_id, weather_condition=Meteo.RAINY
            )

        return check_file
    elif prompt == "A red deer foraging in rainy weather.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            red_deer_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.RED_DEER
            )
            return tracks_contain_activity(
                red_deer_tracks, activity_name=Activity.FORAGING
            ) and check_contains_weather_condition(
                video_id, weather_condition=Meteo.RAINY
            )

        return check_file
    elif prompt == "A juvenile deer in clear or sunny weather.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            return tracks_contain_deer_age(
                individual_tracks,
                age=DAge.JUVENILE,
            ) and (
                check_contains_weather_condition(
                    video_id, weather_condition=Meteo.CLEAR
                )
                or check_contains_weather_condition(
                    video_id, weather_condition=Meteo.SUNNY
                )
            )

        return check_file
    elif prompt == "A juvenile deer in rainy weather.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            return tracks_contain_deer_age(
                individual_tracks,
                age=DAge.JUVENILE,
            ) and check_contains_weather_condition(
                video_id, weather_condition=Meteo.RAINY
            )

        return check_file

    elif prompt == "An adult female red deer escaping in sunny weather.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            female_red_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks, sex=DSex.FEMALE, deer_species=Species.RED_DEER
            )
            return tracks_contain_activity(
                female_red_deer_tracks, activity_name=Activity.ESCAPING
            ) and check_contains_weather_condition(
                video_id, weather_condition=Meteo.SUNNY
            )

        return check_file
    elif prompt == "An animal being vigilant while the weather is rainy or overcast.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            return tracks_contain_activity(
                individual_tracks, activity_name=Activity.VIGILANCE
            ) and (
                check_contains_weather_condition(
                    video_id, weather_condition=Meteo.RAINY
                )
                or check_contains_weather_condition(
                    video_id, weather_condition=Meteo.OVERCAST
                )
            )

        return check_file
    elif prompt == "An animal being vigilant while the weather is clear or sunny.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            return tracks_contain_activity(
                individual_tracks, activity_name=Activity.VIGILANCE
            ) and (
                check_contains_weather_condition(
                    video_id, weather_condition=Meteo.CLEAR
                )
                or check_contains_weather_condition(
                    video_id, weather_condition=Meteo.SUNNY
                )
            )

        return check_file
    elif prompt == "A juvenile red deer suckling.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

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

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

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

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

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

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            return tracks_contain_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION
            )

        return check_file

    elif prompt == "A fox reacting to a camera.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            fox_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.FOX
            )

            return tracks_contain_activity(
                fox_tracks, activity_name=Activity.CAMERA_REACTION
            )

        return check_file

    elif prompt == "A wolf reacting to a camera.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

            wolf_tracks = get_tracks_from_species(
                individual_tracks, species_name=Species.WOLF
            )

            return tracks_contain_activity(
                wolf_tracks, activity_name=Activity.CAMERA_REACTION
            )

        return check_file

    elif prompt == "An animal reacting to a camera and then foraging.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            sequence = [Activity.CAMERA_REACTION, Activity.FORAGING]
            for track in individual_tracks:
                if check_track_contains_continuous_sequence(track, sequence):
                    return True

            return False

        return check_file

    elif (
        prompt
        == "An animal foraging, then reacting to a camera, and then returning to foraging."
    ):

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            sequence = [Activity.FORAGING, Activity.CAMERA_REACTION, Activity.FORAGING]
            for track in individual_tracks:
                if check_track_contains_continuous_sequence(track, sequence):
                    return True

            return False

        return check_file

    elif prompt == "An animal reacting to a camera and then running away.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            sequence = [Activity.CAMERA_REACTION, Action.TROTTING_OR_RUNNING]
            for track in individual_tracks:
                if check_track_contains_continuous_sequence(track, sequence):
                    return True

            return False

        return check_file

    elif (
        prompt
        == "An animal foraging, then reacting to a camera, and then running away."
    ):

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
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
    elif prompt == "An animal reacting to a camera in rainy or overcast weather.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            return tracks_contain_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION
            ) and (
                check_contains_weather_condition(
                    video_id, weather_condition=Meteo.RAINY
                )
                or check_contains_weather_condition(
                    video_id, weather_condition=Meteo.OVERCAST
                )
            )

        return check_file
    elif prompt == "An animal reacting to a camera in clear or sunny weather.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            return tracks_contain_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION
            ) and (
                check_contains_weather_condition(
                    video_id, weather_condition=Meteo.CLEAR
                )
                or check_contains_weather_condition(
                    video_id, weather_condition=Meteo.SUNNY
                )
            )

        return check_file
    elif (
        prompt
        == "An animal escaping from another animal of the same species that is chasing it."
    ):

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

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
        == "An animal escaping from another animal of a different species that is chasing it."
    ):

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

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
        == "Two or more red deer foraging, with at least one also being vigilant at some point."
    ):

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
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

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

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

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

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

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)

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

    elif prompt == "A single adult red deer foraging only.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
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
    elif prompt == "An adult male red deer wallowing.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
            male_red_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.MALE,
                deer_species=Species.RED_DEER,
            )
            return tracks_contain_activity(
                male_red_deer_tracks, activity_name=Activity.MARKING_OR_WALLOWING
            )

        return check_file
    elif (
        prompt
        == "An adult male red deer doing the same sequence of actions while wallowing as any individual of <vid>S3_C3_E545_V0406</vid>."
    ):

        def check_file(video_id):
            video_ref = "S3_C3_E545_V0406"
            individual_tracks_ref = get_tracks_from_id(video_ref)
            male_red_deer_tracks_ref = get_adult_deer_tracks_from_sex(
                individual_tracks_ref,
                sex=DSex.MALE,
                deer_species=Species.RED_DEER,
            )
            male_red_deer_tracks_ref_wallowing = get_tracks_from_activity(
                male_red_deer_tracks_ref, activity_name=Activity.MARKING_OR_WALLOWING
            )

            individual_tracks = get_tracks_from_id(video_id)
            male_red_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.MALE,
                deer_species=Species.RED_DEER,
            )
            male_red_deer_tracks_wallowing = get_tracks_from_activity(
                male_red_deer_tracks, activity_name=Activity.MARKING_OR_WALLOWING
            )

            for track in male_red_deer_tracks_wallowing:
                for ref_track in male_red_deer_tracks_ref_wallowing:
                    ref_sequence = get_action_sequences_from_tracks([ref_track])
                    if check_track_contains_continuous_sequence(track, ref_sequence[0]):
                        return True
            return False

        return check_file

    elif (
        prompt
        == "An adult male red deer doing the same sequence of actions while wallowing as any individual of <vid>S3_C3_E524_V0327</vid>."
    ):

        def check_file(video_id):
            video_ref = "S3_C3_E524_V0327"
            individual_tracks_ref = get_tracks_from_id(video_ref)
            male_red_deer_tracks_ref = get_adult_deer_tracks_from_sex(
                individual_tracks_ref,
                sex=DSex.MALE,
                deer_species=Species.RED_DEER,
            )
            male_red_deer_tracks_ref_wallowing = get_tracks_from_activity(
                male_red_deer_tracks_ref, activity_name=Activity.MARKING_OR_WALLOWING
            )
            individual_tracks = get_tracks_from_id(video_id)
            male_red_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.MALE,
                deer_species=Species.RED_DEER,
            )
            male_red_deer_tracks_wallowing = get_tracks_from_activity(
                male_red_deer_tracks, activity_name=Activity.MARKING_OR_WALLOWING
            )
            for track in male_red_deer_tracks_wallowing:
                for ref_track in male_red_deer_tracks_ref_wallowing:
                    ref_sequence = get_action_sequences_from_tracks([ref_track])
                    if check_track_contains_continuous_sequence(track, ref_sequence[0]):
                        return True
            return False

        return check_file
    elif (
        prompt
        == "An adult male red deer doing the same unique actions while wallowing as any individual of <vid>S3_C3_E524_V0327</vid>."
    ):

        def check_file(video_id):
            video_ref = "S3_C3_E524_V0327"
            individual_tracks_ref = get_tracks_from_id(video_ref)
            individual_tracks = get_tracks_from_id(video_id)
            male_red_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.MALE,
                deer_species=Species.RED_DEER,
            )
            male_red_deer_tracks_ref = get_adult_deer_tracks_from_sex(
                individual_tracks_ref,
                sex=DSex.MALE,
                deer_species=Species.RED_DEER,
            )
            male_red_deer_tracks_wallowing = get_tracks_from_activity(
                male_red_deer_tracks, activity_name=Activity.MARKING_OR_WALLOWING
            )
            male_red_deer_tracks_wallowing_ref = get_tracks_from_activity(
                male_red_deer_tracks_ref, activity_name=Activity.MARKING_OR_WALLOWING
            )
            for track in male_red_deer_tracks_wallowing:
                for ref_track in male_red_deer_tracks_wallowing_ref:
                    ref_actions = get_unique_actions_from_tracks([ref_track])
                    track_actions = get_unique_actions_from_tracks([track])
                    if ref_actions == track_actions:
                        return True
            return False

        return check_file
    elif (
        prompt
        == "An adult male red deer doing the same unique actions while wallowing as any individual of <vid>S3_C2_E524_V0087</vid>."
    ):

        def check_file(video_id):
            video_ref = "S3_C2_E524_V0087"
            individual_tracks_ref = get_tracks_from_id(video_ref)
            individual_tracks = get_tracks_from_id(video_id)
            male_red_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.MALE,
                deer_species=Species.RED_DEER,
            )
            male_red_deer_tracks_ref = get_adult_deer_tracks_from_sex(
                individual_tracks_ref,
                sex=DSex.MALE,
                deer_species=Species.RED_DEER,
            )
            male_red_deer_tracks_wallowing = get_tracks_from_activity(
                male_red_deer_tracks, activity_name=Activity.MARKING_OR_WALLOWING
            )
            male_red_deer_tracks_wallowing_ref = get_tracks_from_activity(
                male_red_deer_tracks_ref, activity_name=Activity.MARKING_OR_WALLOWING
            )
            for track in male_red_deer_tracks_wallowing:
                for ref_track in male_red_deer_tracks_wallowing_ref:
                    ref_actions = get_unique_actions_from_tracks([ref_track])
                    track_actions = get_unique_actions_from_tracks([track])
                    if ref_actions == track_actions:
                        return True
            return False

        return check_file
    elif (
        prompt
        == "An adult male red deer performing at least one different action while wallowing than with any individual of <vid>S3_C3_E524_V0327</vid>."
    ):

        def check_file(video_id):
            video_ref = "S3_C3_E524_V0327"
            individual_tracks_ref = get_tracks_from_id(video_ref)
            individual_tracks = get_tracks_from_id(video_id)
            male_red_deer_tracks = get_adult_deer_tracks_from_sex(
                individual_tracks,
                sex=DSex.MALE,
                deer_species=Species.RED_DEER,
            )
            male_red_deer_tracks_ref = get_adult_deer_tracks_from_sex(
                individual_tracks_ref,
                sex=DSex.MALE,
                deer_species=Species.RED_DEER,
            )
            male_red_deer_tracks_wallowing = get_tracks_from_activity(
                male_red_deer_tracks, activity_name=Activity.MARKING_OR_WALLOWING
            )
            male_red_deer_tracks_wallowing_ref = get_tracks_from_activity(
                male_red_deer_tracks_ref, activity_name=Activity.MARKING_OR_WALLOWING
            )
            for track in male_red_deer_tracks_wallowing:
                for ref_track in male_red_deer_tracks_wallowing_ref:
                    ref_actions = get_unique_actions_from_tracks([ref_track])
                    track_actions = get_unique_actions_from_tracks([track])
                    if len(ref_actions.union(track_actions)) > len(ref_actions):
                        return True
            return False

        return check_file
    elif (
        prompt
        == "A juvenile red deer performing at least one action in common while nursing with any individual in <vid>S1_C6_F404_V0330</vid>."
    ):

        def check_file(video_id):
            video_ref = "S1_C6_F404_V0330"
            individual_tracks_ref = get_tracks_from_id(video_ref)
            individual_tracks = get_tracks_from_id(video_id)
            juvenile_red_deer_tracks = get_deer_tracks_from_age(
                individual_tracks,
                age=DAge.JUVENILE,
                deer_species=Species.RED_DEER,
            )
            juvenile_red_deer_tracks_ref = get_deer_tracks_from_age(
                individual_tracks_ref,
                age=DAge.JUVENILE,
                deer_species=Species.RED_DEER,
            )
            juvenile_red_deer_tracks_nursing = get_tracks_from_activity(
                juvenile_red_deer_tracks, activity_name=Activity.NURSING
            )
            juvenile_red_deer_tracks_nursing_ref = get_tracks_from_activity(
                juvenile_red_deer_tracks_ref, activity_name=Activity.NURSING
            )
            for track in juvenile_red_deer_tracks_nursing:
                for ref_track in juvenile_red_deer_tracks_nursing_ref:
                    ref_actions = get_unique_actions_from_tracks([ref_track])
                    track_actions = get_unique_actions_from_tracks([track])
                    # at least one unique action in track_actions compared to ref_actions
                    if len(track_actions.intersection(ref_actions)) > 0:
                        return True
            return False

        return check_file
    elif (
        prompt
        == "A juvenile red deer performing at least one action in common while nursing with any individual in <vid>S1_C6_F394_V0310</vid>."
    ):

        def check_file(video_id):
            video_ref = "S1_C6_F394_V0310"
            individual_tracks_ref = get_tracks_from_id(video_ref)
            individual_tracks = get_tracks_from_id(video_id)
            juvenile_red_deer_tracks = get_deer_tracks_from_age(
                individual_tracks,
                age=DAge.JUVENILE,
                deer_species=Species.RED_DEER,
            )
            juvenile_red_deer_tracks_ref = get_deer_tracks_from_age(
                individual_tracks_ref,
                age=DAge.JUVENILE,
                deer_species=Species.RED_DEER,
            )
            juvenile_red_deer_tracks_nursing = get_tracks_from_activity(
                juvenile_red_deer_tracks, activity_name=Activity.NURSING
            )
            juvenile_red_deer_tracks_nursing_ref = get_tracks_from_activity(
                juvenile_red_deer_tracks_ref, activity_name=Activity.NURSING
            )
            for track in juvenile_red_deer_tracks_nursing:
                for ref_track in juvenile_red_deer_tracks_nursing_ref:
                    ref_actions = get_unique_actions_from_tracks([ref_track])
                    track_actions = get_unique_actions_from_tracks([track])
                    # at least one unique action in track_actions compared to ref_actions
                    if len(track_actions.intersection(ref_actions)) > 0:
                        return True
            return False

        return check_file
    elif (
        prompt
        == "A juvenile red deer performing at least one different action while nursing than with any individual in <vid>S1_C6_F404_V0330</vid>."
    ):

        def check_file(video_id):
            video_ref = "S1_C6_F404_V0330"
            individual_tracks_ref = get_tracks_from_id(video_ref)
            individual_tracks = get_tracks_from_id(video_id)
            juvenile_red_deer_tracks = get_deer_tracks_from_age(
                individual_tracks,
                age=DAge.JUVENILE,
                deer_species=Species.RED_DEER,
            )
            juvenile_red_deer_tracks_ref = get_deer_tracks_from_age(
                individual_tracks_ref,
                age=DAge.JUVENILE,
                deer_species=Species.RED_DEER,
            )
            juvenile_red_deer_tracks_nursing = get_tracks_from_activity(
                juvenile_red_deer_tracks, activity_name=Activity.NURSING
            )
            juvenile_red_deer_tracks_nursing_ref = get_tracks_from_activity(
                juvenile_red_deer_tracks_ref, activity_name=Activity.NURSING
            )
            for track in juvenile_red_deer_tracks_nursing:
                for ref_track in juvenile_red_deer_tracks_nursing_ref:
                    ref_actions = get_unique_actions_from_tracks([ref_track])
                    track_actions = get_unique_actions_from_tracks([track])
                    if len(ref_actions.union(track_actions)) > len(ref_actions):
                        return True
            return False

        return check_file
    elif (
        prompt
        == "An individual doing the same sequence of actions while reacting to a camera as any individual of <vid>S1_C4_F173_V0137</vid>."
    ):

        def check_file(video_id):
            video_ref = "S1_C4_F173_V0137"
            individual_tracks_ref = get_tracks_from_id(video_ref)
            individual_tracks = get_tracks_from_id(video_id)
            camera_reaction_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION
            )
            camera_reaction_tracks_ref = get_tracks_from_activity(
                individual_tracks_ref, activity_name=Activity.CAMERA_REACTION
            )
            for track in camera_reaction_tracks:
                for ref_track in camera_reaction_tracks_ref:
                    ref_sequences = get_action_sequences_from_tracks([ref_track])
                    if check_track_contains_continuous_sequence(
                        track, ref_sequences[0]
                    ):
                        return True
            return False

        return check_file
    elif (
        prompt
        == "An individual doing the same sequence of actions while reacting to a camera as any individual of <vid>S1_C2_E8_V0021</vid>."
    ):

        def check_file(video_id):
            video_ref = "S1_C2_E8_V0021"
            individual_tracks_ref = get_tracks_from_id(video_ref)
            individual_tracks = get_tracks_from_id(video_id)
            camera_reaction_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION
            )
            camera_reaction_tracks_ref = get_tracks_from_activity(
                individual_tracks_ref, activity_name=Activity.CAMERA_REACTION
            )
            for track in camera_reaction_tracks:
                for ref_track in camera_reaction_tracks_ref:
                    ref_sequences = get_action_sequences_from_tracks([ref_track])
                    if check_track_contains_continuous_sequence(
                        track, ref_sequences[0]
                    ):
                        return True
            return False

        return check_file
    elif (
        prompt
        == "An individual doing the same unique actions while reacting to a camera as any individual of <vid>S1_C2_E8_V0021</vid>."
    ):

        def check_file(video_id):

            video_ref = "S1_C2_E8_V0021"
            individual_tracks_ref = get_tracks_from_id(video_ref)
            individual_tracks = get_tracks_from_id(video_id)
            camera_reaction_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION
            )
            camera_reaction_tracks_ref = get_tracks_from_activity(
                individual_tracks_ref, activity_name=Activity.CAMERA_REACTION
            )
            for track in camera_reaction_tracks:
                for ref_track in camera_reaction_tracks_ref:
                    ref_actions = get_unique_actions_from_tracks([ref_track])
                    track_actions = get_unique_actions_from_tracks([track])
                    if ref_actions == track_actions:
                        return True
            return False

        return check_file
    elif (
        prompt
        == "An individual doing the same unique actions while reacting to a camera as any individual of <vid>S1_C4_F173_V0137</vid>."
    ):

        def check_file(video_id):

            video_ref = "S1_C4_F173_V0137"
            individual_tracks_ref = get_tracks_from_id(video_ref)
            individual_tracks = get_tracks_from_id(video_id)
            camera_reaction_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION
            )
            camera_reaction_tracks_ref = get_tracks_from_activity(
                individual_tracks_ref, activity_name=Activity.CAMERA_REACTION
            )
            for track in camera_reaction_tracks:
                for ref_track in camera_reaction_tracks_ref:
                    ref_actions = get_unique_actions_from_tracks([ref_track])
                    track_actions = get_unique_actions_from_tracks([track])
                    if ref_actions == track_actions:
                        return True
            return False

        return check_file
    elif (
        prompt
        == "An individual doing at least one different action while reacting to a camera than with any individual of <vid>S1_C2_E8_V0021</vid>."
    ):

        def check_file(video_id):
            video_ref = "S1_C2_E8_V0021"
            individual_tracks_ref = get_tracks_from_id(video_ref)
            individual_tracks = get_tracks_from_id(video_id)
            camera_reaction_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION
            )
            camera_reaction_tracks_ref = get_tracks_from_activity(
                individual_tracks_ref, activity_name=Activity.CAMERA_REACTION
            )
            for track in camera_reaction_tracks:
                track_actions = get_unique_actions_from_tracks([track])
                for ref_track in camera_reaction_tracks_ref:
                    ref_actions = get_unique_actions_from_tracks([ref_track])
                    if len(ref_actions.union(track_actions)) > len(ref_actions):
                        return True
            return False

        return check_file
    elif (
        prompt
        == "An individual doing at least one different action while reacting to a camera than with any individual of <vid>S1_C4_F173_V0137</vid>."
    ):

        def check_file(video_id):
            video_ref = "S1_C4_F173_V0137"
            individual_tracks_ref = get_tracks_from_id(video_ref)
            individual_tracks = get_tracks_from_id(video_id)
            camera_reaction_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.CAMERA_REACTION
            )
            camera_reaction_tracks_ref = get_tracks_from_activity(
                individual_tracks_ref, activity_name=Activity.CAMERA_REACTION
            )
            for track in camera_reaction_tracks:
                track_actions = get_unique_actions_from_tracks([track])
                for ref_track in camera_reaction_tracks_ref:
                    ref_actions = get_unique_actions_from_tracks([ref_track])
                    if len(ref_actions.union(track_actions)) > len(ref_actions):
                        return True
            return False

        return check_file
    elif (
        prompt
        == "An animal being vigilant in a different weather condition from that in <vid>S2_C2_F536_V0066</vid>."
    ):

        def check_file(video_id):
            video_ref = "S2_C2_F536_V0066"
            individual_tracks = get_tracks_from_id(video_id)
            vigilance_tracks = get_tracks_from_activity(
                individual_tracks, activity_name=Activity.VIGILANCE
            )
            individual_tracks_ref = get_tracks_from_id(video_ref)
            vigilance_tracks_ref = get_tracks_from_activity(
                individual_tracks_ref, activity_name=Activity.VIGILANCE
            )
            weather_condition = get_weather_conditions_from_videos(video_id)
            weather_condition_ref = get_weather_conditions_from_videos(video_ref)
            if len(vigilance_tracks) > 0 and len(vigilance_tracks_ref) > 0:
                return weather_condition != weather_condition_ref
            return False

        return check_file
    elif (
        prompt
        == "A wolf chasing prey in a different weather condition from that in <vid>S2_C1_F573_V0093</vid>."
    ):

        def check_file(video_id):
            video_ref = "S2_C1_F573_V0093"
            individual_tracks = get_tracks_from_id(video_id)
            wolf_tracks = get_tracks_from_species(individual_tracks, Species.WOLF)
            wolf_chasing_tracks = get_tracks_from_activity(
                wolf_tracks, activity_name=Activity.CHASING
            )
            if len(wolf_chasing_tracks) == 0:
                return False
            weather_condition = get_weather_conditions_from_videos(video_id)
            weather_condition_ref = get_weather_conditions_from_videos(video_ref)
            return weather_condition != weather_condition_ref

        return check_file
    elif (
        prompt
        == "An individual sharing at least one activity with any individual from <vid>S2_C1_F573_V0093</vid>."
    ):

        def check_file(video_id):
            video_ref = "S2_C1_F573_V0093"
            if not video_ref:
                return False
            individual_tracks_ref = get_tracks_from_id(video_ref)
            individual_tracks = get_tracks_from_id(video_id)
            for track in individual_tracks:
                track_activities = get_unique_activities_from_tracks([track])
                for ref_track in individual_tracks_ref:
                    ref_track_activities = get_unique_activities_from_tracks(
                        [ref_track]
                    )
                    if len(track_activities.intersection(ref_track_activities)) > 0:
                        return True
            return False

        return check_file
    elif (
        prompt
        == "An individual sharing at least one activity with any individual from <vid>S3_C2_E670_V0159</vid> but in a different weather condition."
    ):

        def check_file(video_id):
            video_ref = "S3_C2_E670_V0159"
            individual_tracks_ref = get_tracks_from_id(video_ref)
            individual_tracks = get_tracks_from_id(video_id)
            weather_condition = get_weather_conditions_from_videos(video_id)
            weather_condition_ref = get_weather_conditions_from_videos(video_ref)
            for track in individual_tracks:
                track_activities = get_unique_activities_from_tracks([track])
                for ref_track in individual_tracks_ref:
                    ref_track_activities = get_unique_activities_from_tracks(
                        [ref_track]
                    )
                    if (
                        len(track_activities.intersection(ref_track_activities)) > 0
                        and weather_condition != weather_condition_ref
                    ):
                        return True
            return False

        return check_file
    elif (
        prompt
        == "An individual sharing at least one activity with any individual of the same species from <vid>S1_C4_F136_V0115</vid> but in a different weather condition."
    ):

        def check_file(video_id):
            video_ref = "S1_C4_F136_V0115"
            individual_tracks_ref = get_tracks_from_id(video_ref)
            individual_tracks = get_tracks_from_id(video_id)
            weather_condition = get_weather_conditions_from_videos(video_id)
            weather_condition_ref = get_weather_conditions_from_videos(video_ref)
            for track in individual_tracks:
                track_activities = get_unique_activities_from_tracks([track])
                species_vid = get_unique_species_from_tracks([track])
                for ref_track in individual_tracks_ref:
                    species_ref = get_unique_species_from_tracks([ref_track])
                    ref_track_activities = get_unique_activities_from_tracks(
                        [ref_track]
                    )
                    if (
                        len(track_activities.intersection(ref_track_activities)) > 0
                        and weather_condition != weather_condition_ref
                        and species_vid == species_ref
                    ):
                        return True
            return False

        return check_file
    elif (
        prompt
        == "An individual sharing at least one activity with any individual from <vid>S1_C1_E66_V0152</vid> but of a different species."
    ):

        def check_file(video_id):
            video_ref = "S1_C1_E66_V0152"
            individual_tracks_ref = get_tracks_from_id(video_ref)
            individual_tracks = get_tracks_from_id(video_id)
            weather_condition = get_weather_conditions_from_videos(video_id)
            weather_condition_ref = get_weather_conditions_from_videos(video_ref)
            for track in individual_tracks:
                track_activities = get_unique_activities_from_tracks([track])
                species_vid = get_unique_species_from_tracks([track])
                for ref_track in individual_tracks_ref:
                    species_ref = get_unique_species_from_tracks([ref_track])
                    ref_track_activities = get_unique_activities_from_tracks(
                        [ref_track]
                    )
                    if (
                        len(track_activities.intersection(ref_track_activities)) > 0
                        and weather_condition != weather_condition_ref
                        and species_vid != species_ref
                    ):
                        return True
            return False

        return check_file
    elif (
        prompt
        == "An individual performing the same activities as any individual from <vid>S1_C2_E179_V0409</vid> but in a different weather condition."
    ):

        def check_file(video_id):
            video_ref = "S1_C2_E179_V0409"
            individual_tracks_ref = get_tracks_from_id(video_ref)
            individual_tracks = get_tracks_from_id(video_id)
            weather_condition = get_weather_conditions_from_videos(video_id)
            weather_condition_ref = get_weather_conditions_from_videos(video_ref)
            for track in individual_tracks:
                track_activities = get_unique_activities_from_tracks([track])
                for ref_track in individual_tracks_ref:
                    ref_track_activities = get_unique_activities_from_tracks(
                        [ref_track]
                    )
                    if (
                        track_activities == ref_track_activities
                        and weather_condition != weather_condition_ref
                    ):
                        return True
            return False

        return check_file
    elif prompt == "An empty video.":

        def check_file(video_id):
            individual_tracks = get_tracks_from_id(video_id)
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
    parser.add_argument(
        "-OJ",
        "--output_json_file",
        type=str,
        default="./queries_and_videos.json",
        help="Path to the output dictionnary containing association results.",
    )
    args = parser.parse_args()
    json_folder = Path(args.json_folder)

    with open("queries_and_videos_empty.json", "r") as f:
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
                    if parsing_functions[q](f.stem):
                        out_queries_dict[q_cat][q].append(f.stem)
                except Exception as e:
                    print(f"Could not parse {f} for {q}")
                    print(e)
                    continue

    # Print and save results
    for q_cat in queries_dict:
        for q in queries_dict[q_cat]:
            print(q, ":", len(out_queries_dict[q_cat][q]), "videos")

    with open(args.output_json_file, "w") as f:
        json.dump(out_queries_dict, f, indent=2)
