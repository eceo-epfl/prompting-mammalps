import argparse
import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Literal

from smolagents import tool

FrameDetections = Dict
VideoDict = List[FrameDetections]
InfoDict = Dict

BehaviorSegment = List[Dict]
IndividualTrack = List[BehaviorSegment]

class Species(Enum):
    RED_DEER = "red_deer"
    ROE_DEER = "roe_deer"
    FOX = "fox"
    HARE = "hare"
    MARTEN = "marten"
    WOLF = "wolf"
    CHAMOIS = "chamois"


class Action(Enum):
    WALKING = "walking"
    STANDING_HEAD_UP = "standing_head_up"
    STANDING_HEAD_DOWN = "standing_head_down"
    GRAZING = "grazing"
    SNIFFING = "sniffing"
    LOOKING_AT_CAMERA = "looking_at_camera"
    TROTTING_OR_RUNNING = "trotting_or_running"
    SCRATCHING_OWN_HEAD_OR_BODY = "scratching_own_head_or_body"
    RUBBING_ANTLERS_ON_GROUND = "rubbing_antlers_on_ground"
    PAWING_GROUND = "pawing_ground"
    SHAKING_HEAD_OR_BODY = "shaking_head_or_body"
    VOCALIZING = "vocalizing"
    BATHING = "bathing"
    JUMPING = "jumping"
    DRINKING = "drinking"
    LAYING_DOWN = "laying_down"
    DEFECATING = "defecating"
    URINATING = "urinating"
    BROWSING = "browsing"
    STRETCHING_BODY = "stretching_body"
    SUCKLING = "suckling"
    PREPARING_TO_SUCKLE = "preparing_to_suckle"


class Activity(Enum):
    FORAGING = "foraging"
    VIGILANCE = "vigilance"
    COURTSHIP = "courtship"
    CAMERA_REACTION = "camera_reaction"
    ESCAPING = "escaping"
    CHASING = "chasing"
    NURSING = "nursing"
    GROOMING = "grooming"
    PLAYING = "playing"
    RESTING = "resting"
    MARKING_OR_WALLOWING = "marking_or_wallowing"


class DAge(Enum):
    """Deer age"""

    ADULT = "adult"
    JUVENILE = "juvenile"


class DSex(Enum):
    "Sex for adult deers"

    MALE = "male"
    FEMALE = "female"


class Meteo(Enum):
    SUNNY = "sunny"
    CLEAR = "clear"
    OVERCAST = "overcast"
    RAINY = "rainy"


# Global variable for JSON annotations folder
# Necessary to avoid passing this as parameter to functions
# which could be changed by the code agent
JSON_FOLDER = Path("/media/EVO870/datasets/prompting-mammalps-v2/annotations")


### Basic functions
def load_video_detections(json_file: Union[Path, str]) -> VideoDict:
    """
    Loads video detections from a JSON file.
    Args:
        json_file (Union[Path, str]): The path to the JSON file.
    Returns:
        VideoDict: A dictionary containing video frame detections.
    """
    with open(json_file, "r") as f:
        content = json.load(f)

    return content["frames"]


def load_video_info(json_file: Union[Path, str]) -> InfoDict:
    with open(json_file, "r") as f:
        content = json.load(f)

    return content["info"]


def get_tracks_from_json(json_file: str) -> List[IndividualTrack]:
    """
    Retrieves all individual tracks from a given JSON file.
    Args:
        json_file (str): input json file

    Returns:
        List[IndividualTracks]: List of individual tracks present in the JSON file.
    """

    video_detections = load_video_detections(json_file)
    individual_tracks = get_tracks_from_video_detections(video_detections)

    return individual_tracks

@tool
def get_tracks_from_id(
    video_id: Union[Path, str],
) -> List[IndividualTrack]:
    video_id_str = str(video_id)
    json_file_path = str(next(JSON_FOLDER.rglob(f"*/{video_id_str}.json")))
    video_detections = load_video_detections(json_file_path)
    individual_tracks = get_tracks_from_video_detections(video_detections)
    return individual_tracks

## Weather and time attributes
@tool
def get_weather_conditions_from_videos(video_id: Union[Path, str]) -> Meteo:
    """Retrieve weather from video info attributes weather conditions
    Args:
        video_id (Union[Path, str]): The ID of the video.
    Returns:
        Meteo: The weather condition of the video.
    """
    video_id_str = str(video_id)
    json_file_path = str(next(JSON_FOLDER.rglob(f"*/{video_id_str}.json")))
    video_info = load_video_info(json_file_path)
    return video_info["attributes"]["weather"]

@tool
def check_contains_weather_condition(
    video_id: Union[Path, str], weather_condition: Meteo
) -> bool:
    """Check if the video contains the specified weather condition
    Args:
        json_file (Union[Path, str]): The path to the JSON file.
        weather_condition (Meteo): The weather condition to check.
    Returns:
        bool: True if the video contains the specified weather condition, False otherwise.
    """
    video_weather = get_weather_conditions_from_videos(video_id)
    return video_weather == weather_condition

def get_tracks_from_video_detections(
    video_detections: VideoDict,
) -> List[IndividualTrack]:
    """Organizes video detections into individual tracks grouped by segment.
    This function processes a list of frame detections and organizes them into individual
    tracks, where each track contains behavior segments as a list of detections displaying the same animal behavior
    for a specific individual.
    Args:
        video_detections (VideoDict): A list of frame detections, where each frame contains
            a list of detection dictionaries with 'track_id', 'segment_id', and other detection
            information.
    Returns:
        List[IndividualTrack]: A list of individual tracks, where each track is organized by
            segments containing lists of detections.
    Example structure:
        Input video_detections:
        [
            {
                "frame_id": 1,
                "detections": [
                    {"track_id": 1, "segment_id": 1, ...},
                    {"track_id": 2, "segment_id": 1, ...}
                ]
            },
            ...
        ]
        Output:
        [
            [  # Individual 1
                [detection1, detection2, ..., detection7],  # BehaviorSegment 1
                [detection30, detection31, ..., detection45]   # BehaviorSegment 2
            ],
            ...  # More individuals
        ]
    """
    individual_tracks = {}
    prev_attrs = {}
    beh_seg_id = 0

    # Parse detections
    for frame_detections in video_detections:
        for detection in frame_detections["detections"]:
            individual_id = detection["track_id"]
            detection["frame_id"] = frame_detections["frame_id"]

            # Check if there is a change in the individual behavior
            # If so, create a new behavior segment
            if (
                individual_id not in prev_attrs
                or detection.get("attributes") != prev_attrs[individual_id]
            ):
                beh_seg_id += 1

            prev_attrs[individual_id] = detection.get("attributes")

            if individual_id in individual_tracks:
                if beh_seg_id in individual_tracks[individual_id]:
                    individual_tracks[individual_id][beh_seg_id].append(detection)
                else:
                    individual_tracks[individual_id][beh_seg_id] = [detection]
            else:
                individual_tracks[individual_id] = {beh_seg_id: [detection]}

    # Return list of list instead of dict
    for i, it in individual_tracks.items():
        individual_tracks[i] = [s for s in it.values()]
    individual_tracks = [it for it in individual_tracks.values()]

    return individual_tracks


def check_segment_contains_attribute(
    segment: BehaviorSegment,
    attribute_name: str,
    attribute_value: Union[Species, Action, Activity, DSex, DAge, Meteo],
):
    # Checks if the first element of the segment has the given attribute name and value
    expected_value = (
        attribute_value.value if isinstance(attribute_value, Enum) else attribute_value
    )
    return (
        "attributes" in segment[0]
        and segment[0]["attributes"].get(attribute_name) == expected_value
    )


def check_segment_contains_any_attribute_value(
    segment: BehaviorSegment,
    attribute_value: Union[Species, Action, Activity, DSex, DAge, Meteo],
):
    # Checks if the first element of the segment has the given attribute value for Action, Action2, or Activity
    return (
        check_segment_contains_attribute(segment, "Action", attribute_value)
        or check_segment_contains_attribute(segment, "Action2", attribute_value)
        or check_segment_contains_attribute(segment, "Activity", attribute_value)
    )


def get_tracks_from_attribute(
    individual_tracks: List[IndividualTrack],
    attribute_name: str,
    attribute_value: Union[Species, Action, Activity, DSex, DAge, Meteo],
):
    """
    Retrieve tracks corresponding to a given attribute name and value.
    All behavior segments are returned even if some don't match the given attribute.

    Args:
        individual_tracks (List[IndividualTrack]): List of individual tracks, each organized by segments.
        attribute_name (str): The attribute name to match.
        attribute_value (str): The attribute value to match.

    Returns:
        List[IndividualTrack]: List of tracks where at least one detection has the specified attribute value.
    """
    attr_tracks = []

    for track in individual_tracks:
        found = False
        for segment in track:
            if check_segment_contains_attribute(
                segment, attribute_name, attribute_value
            ):
                attr_tracks.append(track)
                found = True
            if found:
                break  # Stop processing this track after first match

    return attr_tracks


def get_segments_from_attribute(
    individual_tracks: List[IndividualTrack],
    attribute_name: str,
    attribute_value: Union[Species, Action, Activity, DSex, DAge, Meteo],
):
    """
    Retrieve behavior segments corresponding to a given attribute name and value.

    Args:
        individual_tracks (List[IndividualTrack]): List of individual tracks, each organized by segments.
        attribute_name (str): The attribute name to match.
        attribute_value (str): The attribute value to match.

    Returns:
        List[BehaviorSegment]: List of behavior segments where detections contain the specified attribute value.
    """

    return [
        segment
        for track in individual_tracks
        for segment in track
        if check_segment_contains_attribute(segment, attribute_name, attribute_value)
    ]


def get_segments_from_attribute_as_tracks(
    individual_tracks: List[IndividualTrack],
    attribute_name: str,
    attribute_value: Union[Species, Action, Activity, DSex, DAge, Meteo],
):
    """
    Retrieve tracks corresponding to a given attribute name and value.
    Only behavior segments matching the given attribute are returned.

    Args:
        individual_tracks (List[IndividualTrack]): List of individual tracks, each organized by segments.
        attribute_name (str): The attribute name to match.
        attribute_value (str): The attribute value to match.

    Returns:
        List[IndividualTrack]: List of tracks where at least one detection has the specified attribute value.
    """
    attr_tracks = []

    for track in individual_tracks:
        track = [
            segment
            for segment in track
            if check_segment_contains_attribute(
                segment, attribute_name, attribute_value
            )
        ]
        if len(track) > 0:
            attr_tracks.append(track)

    return attr_tracks

def check_enum_type(value, enumType, allow_none: bool = False):
    if allow_none and value is None:
        return
    if not isinstance(value, enumType):
        print(f"{value} must be an element from {enumType}")
        print(f"Available {enumType} are:", [e for e in enumType])
        raise AttributeError

@tool
def get_tracks_from_species(
    individual_tracks: List[IndividualTrack], species_name: Literal[Species]
) -> List[IndividualTrack]:
    """
    Retrieve tracks corresponding to a given species name.
    Args:
        individual_tracks (List[IndividualTrack]): a list of individual tracks
        species_name (Species): the species name to retrieve tracks from.

    Returns:
        List[IndividualTrack]: The reduced list of individual tracks where only tracks corresponding to the given species_name have been preserved
    """
    check_enum_type(species_name, Species)

    return get_tracks_from_attribute(individual_tracks, "Species", species_name)


@tool
def get_tracks_from_action(
    individual_tracks: List[IndividualTrack], action_name: Literal[Action]
) -> List[IndividualTrack]:
    """
    Retrieve tracks corresponding to a given action name.
    Args:
        individual_tracks (List[IndividualTrack]): a list of individual tracks
        action_name (Action): the action name to retrieve tracks from.

    Returns:
        List[IndividualTrack]: The reduced list of individual tracks where only tracks corresponding to the given action_name have been preserved
    """
    check_enum_type(action_name, Action)
    attr_tracks = get_segments_from_attribute_as_tracks(
        individual_tracks, "Action", action_name
    ) + get_segments_from_attribute_as_tracks(individual_tracks, "Action2", action_name)

    return attr_tracks  # duplicates if action = action2


@tool
def get_tracks_from_activity(
    individual_tracks: List[IndividualTrack], activity_name: Literal[Activity]
) -> List[IndividualTrack]:
    """
    Retrieve tracks corresponding to a given activity name.
    Args:
        individual_tracks (List[IndividualTrack]): a list of individual tracks
        activity_name (Activity): the activity name to retrieve tracks from.

    Returns:
        List[IndividualTrack]: The reduced list of individual tracks where only tracks corresponding to the given activity_name have been preserved
    """
    check_enum_type(activity_name, Activity)
    return get_segments_from_attribute_as_tracks(
        individual_tracks, "Activity", activity_name
    )


@tool
def get_deer_tracks_from_age(
    individual_tracks: List[IndividualTrack],
    age: Literal[DAge],
    deer_species: Optional[Species] = None,
) -> List[IndividualTrack]:
    """
    Retrieve deer tracks corresponding to a given age group.
    Args:
        individual_tracks (List[IndividualTrack]): a list of individual tracks
        age (DAge): deer age group to retrieve
        deer_species (Optional[Species]): the deer species name to retrieve tracks from. If None, applies to both red deer and roe deer.

    Returns:
        List[IndividualTrack]: The reduced list of individual tracks where only deer tracks corresponding to the given age group have been preserved
    """
    check_enum_type(age, DAge)
    check_enum_type(deer_species, Species, allow_none=True)
    if deer_species is not None:
        deer_tracks = get_tracks_from_species(
            individual_tracks, species_name=deer_species
        )
    else:
        deer_tracks = get_tracks_from_species(
            individual_tracks, Species.RED_DEER
        ) + get_tracks_from_species(individual_tracks, Species.ROE_DEER)
    return get_tracks_from_attribute(deer_tracks, "Deer_age", age)


@tool
def get_adult_deer_tracks_from_sex(
    individual_tracks: List[IndividualTrack],
    sex: Literal[DSex],
    deer_species: Optional[Species] = None,
) -> List[IndividualTrack]:
    """
    Retrieve adult deer tracks corresponding to a given sex group.
    Args:
        individual_tracks (List[IndividualTrack]): a list of individual tracks
        sex (DSex): adult deer sex group to retrieve
        deer_species (Optional[Species]): the deer species name to retrieve tracks from. If None, applies to both red deer and roe deer.

    Returns:
        List[IndividualTrack]: The reduced list of individual tracks where only adult deer tracks corresponding to the given sex group have been preserved
    """
    check_enum_type(sex, DSex)
    check_enum_type(deer_species, Species, allow_none=True)

    adult_deer_tracks = get_deer_tracks_from_age(
        individual_tracks, DAge.ADULT, deer_species=deer_species
    )
    return get_tracks_from_attribute(adult_deer_tracks, "Deer_adult_sex", sex)


@tool
def get_nb_tracks_species_in_video(
    individual_tracks: List[IndividualTrack], species_name: Literal[Species]
) -> int:
    """
    Returns the number of tracks for a given species in the video
    Args:
        individual_tracks (List[IndividualTrack]): a list of individual tracks
        species_name (Species): the species to count the number of tracks of

    Returns:
        int: the number of individual tracks corresponding to the given species.
    """
    check_enum_type(species_name, Species)

    return len(get_tracks_from_species(individual_tracks, species_name))


@tool
def tracks_contain_species(
    individual_tracks: List[IndividualTrack], species_name: Literal[Species]
) -> bool:
    """
    Checks if any of the individual tracks contain the species of interest.
    Args:
        individual_tracks (List[IndividualTrack]): a list of individual tracks
        species_name (Species): the species to count the number of tracks of

    Returns:
        bool: True if at least one of the individual tracks contain the given species.
    """
    check_enum_type(species_name, Species)

    return get_nb_tracks_species_in_video(individual_tracks, species_name) >= 1


@tool
def get_nb_tracks_action_in_video(
    individual_tracks: List[IndividualTrack], action_name: Literal[Action]
) -> int:
    """
    Returns the number of tracks for a given action in the video
    Args:
        individual_tracks (List[IndividualTrack]): a list of individual tracks
        action_name (Action): the action to count the number of tracks of

    Returns:
        int: the number of individual tracks contaning at least one BehaviorSegment of the given action_name.
    """
    check_enum_type(action_name, Action)
    return len(get_tracks_from_action(individual_tracks, action_name))


@tool
def tracks_contain_action(
    individual_tracks: List[IndividualTrack], action_name: Literal[Action]
) -> bool:
    """
    Checks if any of the individual tracks contain the action of interest.
    Args:
        individual_tracks (List[IndividualTrack]): a list of individual tracks
        action_name (Action): the action to count the number of tracks of

    Returns:
        bool: True if the video contains at least one BehaviorSegment of the given action_name
    """
    check_enum_type(action_name, Action)
    return get_nb_tracks_action_in_video(individual_tracks, action_name) >= 1


@tool
def get_nb_tracks_activity_in_video(
    individual_tracks: List[IndividualTrack], activity_name: Literal[Activity]
) -> int:
    """
    Returns the number of tracks for a given activity in the video
    Args:
        individual_tracks (List[IndividualTrack]): a list of individual tracks
        activity_name (Activity): the activity to count the number of tracks of

    Returns:
        int: the number of individual tracks contaning at least one BehaviorSegment of the given activity_name.
    """
    check_enum_type(activity_name, Activity)
    return len(get_tracks_from_activity(individual_tracks, activity_name))


@tool
def tracks_contain_activity(
    individual_tracks: List[IndividualTrack],
    activity_name: Literal[Activity],
) -> bool:
    """
    Checks if any of the individual tracks contain the activity of interest.
    Args:
        individual_tracks (List[IndividualTrack]): a list of individual tracks
        activity_name (Activity): the activity to count the number of tracks of

    Returns:
        bool: True if the video contains at least one BehaviorSegment of the given activity_name
    """
    check_enum_type(activity_name, Activity)
    return get_nb_tracks_activity_in_video(individual_tracks, activity_name) >= 1


@tool
def get_nb_deer_tracks_age_in_video(
    individual_tracks: List[IndividualTrack],
    age: Literal[DAge],
    deer_species: Optional[Species] = None,
) -> int:
    """
    Retrieve deer tracks corresponding to a given age group.
    Args:
        individual_tracks (List[IndividualTrack]): a list of individual tracks
        age (DAge): deer age group to retrieve
        deer_species (Optional[Species]): the deer species name to retrieve tracks from. If None, applies to both red deer and roe deer.

    Returns:
        int: The number of individual tracks containing a deer of the given age group.
    """
    check_enum_type(age, DAge)
    check_enum_type(deer_species, Species, allow_none=True)
    return len(
        get_deer_tracks_from_age(individual_tracks, age, deer_species=deer_species)
    )


@tool
def tracks_contain_deer_age(
    individual_tracks: List[IndividualTrack],
    age: Literal[DAge],
    deer_species: Optional[Species] = None,
) -> bool:
    """
    Checks if any of the individual tracks contain a deer of the given age group.
    Args:
        individual_tracks (List[IndividualTrack]): a list of individual tracks
        age (DAge): deer age group to retrieve
        deer_species (Optional[Species]): the deer species name to retrieve tracks from. If None, applies to both red deer and roe deer.

    Returns:
        bool: True if the video contains at least one deer track corresponding to the given age group
    """
    check_enum_type(age, DAge)
    check_enum_type(deer_species, Species, allow_none=True)
    return (
        get_nb_deer_tracks_age_in_video(
            individual_tracks, age, deer_species=deer_species
        )
        >= 1
    )


@tool
def get_nb_adult_deer_tracks_sex_in_video(
    individual_tracks: List[IndividualTrack],
    sex: Literal[DSex],
    deer_species: Optional[Species] = None,
) -> int:
    """
    Retrieve adult deer tracks corresponding to a given sex group.
    Args:
        individual_tracks (List[IndividualTrack]): a list of individual tracks
        sex (DSex): adult deer sex group to retrieve
        deer_species (Optional[Species]): the deer species name to retrieve tracks from. If None, applies to both red deer and roe deer.

    Returns:
        int: The number of individual tracks containing an adult deer of the given sex group.
    """
    check_enum_type(sex, DSex)
    check_enum_type(deer_species, Species, allow_none=True)
    return len(
        get_adult_deer_tracks_from_sex(
            individual_tracks, sex, deer_species=deer_species
        )
    )


@tool
def tracks_contain_adult_deer_sex(
    individual_tracks: List[IndividualTrack],
    sex: Literal[DSex],
    deer_species: Optional[Species] = None,
) -> bool:
    """
    Checks if any of the individual tracks contain an adult deer of the given sex group.
    Args:
        individual_tracks (List[IndividualTrack]): a list of individual tracks
        sex (DSex): adult deer sex group to retrieve
        deer_species (Optional[Species]): the deer species name to retrieve tracks from. If None, applies to both red deer and roe deer.

    Returns:
        bool: True if the video contains at least one adult deer track corresponding to the given sex group
    """
    check_enum_type(sex, DSex)
    check_enum_type(deer_species, Species, allow_none=True)
    return (
        get_nb_adult_deer_tracks_sex_in_video(
            individual_tracks, sex, deer_species=deer_species
        )
        >= 1
    )


@tool
def get_unique_species_from_tracks(
    individual_tracks: List[IndividualTrack],
) -> set:
    """
    Returns the set of species present in the individual tracks

    Args:
        individual_tracks (List[IndividualTrack]): List of individual tracks, each organized by segments.

    Returns:
        set[Species]: the unique species present in the individual tracks
    """

    # Since species are not mutable attributes, it is consistent in a track and we can retrieve first elements only
    return set(
        track[0][0]["attributes"]["Species"]
        for track in individual_tracks
        if ("attributes" in track[0][0] and "Species" in track[0][0]["attributes"])
    )


@tool
def get_unique_actions_from_tracks(
    individual_tracks: List[IndividualTrack],
) -> set:
    """
    Returns the set of actions present in the individual tracks

    Args:
        individual_tracks (List[IndividualTrack]): List of individual tracks, each organized by segments.

    Returns:
        set[Action]: the unique actions present in the individual tracks
    """
    actions = []
    for track in individual_tracks:
        for segment in track:
            if "attributes" in segment[0]:
                if "Action" in segment[0]["attributes"]:
                    actions.append(segment[0]["attributes"]["Action"])
                if (
                    "Action2" in segment[0]["attributes"]
                    and segment[0]["attributes"]["Action2"] != "none"
                ):
                    actions.append(segment[0]["attributes"]["Action2"])

    return set(actions)


@tool
def get_unique_activities_from_tracks(
    individual_tracks: List[IndividualTrack],
) -> set:
    """
    Returns the set of activities present in the individual tracks

    Args:
        individual_tracks (List[IndividualTrack]): List of individual tracks, each organized by segments.

    Returns:
        set[Activity]: the unique activities present in the individual tracks
    """

    return set(
        segment[0]["attributes"]["Activity"]
        for track in individual_tracks
        for segment in track
        if ("attributes" in segment[0] and "Activity" in segment[0]["attributes"])
    )


@tool
def check_track_contains_continuous_sequence(
    single_track: IndividualTrack,
    attributes_sequence: Union[
        List[Union[Action, Activity]], List[Action], List[Activity]
    ],
) -> bool:
    """
    Checks if a given individual track contains a continuous series of behavior segments that contain attributes matching the sequence of interest
    Args:
        single_track (IndividualTrack): an individual track
        attributes_sequence (List[Union[Action, Activity]]): an ordered list representing a sequence of actions or activities

    Returns:
        bool: True if the entire sequence appears in the track, False otherwise. Returns True if an empty sequence is given
    """
    if not attributes_sequence:
        return True

    for attribute in attributes_sequence:
        if isinstance(attribute, Action):
            check_enum_type(attribute, Action)
        elif isinstance(attribute, Activity):
            check_enum_type(attribute, Activity)
        else:
            raise AttributeError(f"{attribute} must be an element from either an Action or an Activity")

    # Get anchor segment matching first sequence element
    for bs_id, behavior_segment in enumerate(single_track):
        seq_iter = iter(attributes_sequence)
        next_elem = next(seq_iter)

        if check_segment_contains_any_attribute_value(behavior_segment, next_elem):
            prev_elem = next_elem
            for next_elem in seq_iter:
                # We skip following segments if they still match the previous attribute
                # With the exception of the next element being present
                while (
                    bs_id < len(single_track)
                    and check_segment_contains_any_attribute_value(
                        single_track[bs_id], prev_elem
                    )
                    and (
                        not check_segment_contains_any_attribute_value(
                            single_track[bs_id], next_elem
                        )
                    )
                ):
                    bs_id += 1
                if bs_id >= len(single_track):
                    # We reached the end of the track, sequence can't be completed
                    break
                if not check_segment_contains_any_attribute_value(
                    single_track[bs_id], next_elem
                ):
                    # The next behavior segment that does not match previous attribute also does not match the next one
                    # So the sequence is not complete.
                    break
                prev_elem = next_elem
            else:
                return True

    return False

@tool
def get_action_sequences_from_tracks(
    individual_tracks: List[IndividualTrack],
) -> set:
    """
    Returns the set of unique action sequences present in the individual tracks

    Args:
        individual_tracks (List[IndividualTrack]): List of individual tracks, each organized by segments.

    Returns:
        set: the unique action sequences present in the individual tracks
    """
    action_sequence_tracks = []
    for track in individual_tracks:
        track_actions = []
        for segment in track:
            if "attributes" in segment[0]:
                if "Action" in segment[0]["attributes"]:
                    if (
                        len(track_actions) == 0
                        or track_actions[-1] != segment[0]["attributes"]["Action"]
                    ):
                        track_actions.append(segment[0]["attributes"]["Action"])
                if (
                    "Action2" in segment[0]["attributes"]
                    and segment[0]["attributes"]["Action2"] != "none"
                ):
                    if (
                        len(track_actions) == 0
                        or track_actions[-1] != segment[0]["attributes"]["Action2"]
                    ):
                        track_actions.append(segment[0]["attributes"]["Action2"])
        action_sequence_tracks.append(track_actions)
    return action_sequence_tracks