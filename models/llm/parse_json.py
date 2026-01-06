import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

FrameDetections = Dict
VideoDict = List[FrameDetections]
InfoDict = Dict

Segment = List[Dict]
IndividualTrack = List[Segment]

Point = Tuple[int]
BBox = Tuple[Point]
Mask = List[Point]


### Basic functions
def load_video_detections(json_file: Union[Path, str]) -> VideoDict:
    """Load the frame detections given a json file"""
    with open(json_file, "r") as f:
        content = json.load(f)

    return content["frames"]

def load_video_info(json_file: Union[Path, str]) -> InfoDict:
    """Load video metadata given the json file"""
    with open(json_file, "r") as f:
        content = json.load(f)

    return content["info"]

def get_tracks_from_video_detections(
    video_detections: VideoDict,
) -> List[IndividualTrack]:
    """Organizes video detections into individual tracks grouped by segment.
    This function processes a list of frame detections and organizes them into individual
    tracks, where each track contains detections for a specific individual across different
    segments.
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
                [detection1, detection2, ..., detection7],  # Segment 1
                [detection30, detection31, ..., detection45]   # Segment 2
            ],
            ...  # More individuals
        ]
    """
    individual_tracks = {}

    # Parse detections
    for frame_detections in video_detections:
        for detection in frame_detections["detections"]:
            individual_id = detection["track_id"]
            segment_id = detection.get("segment_id", 1)
            detection["frame_id"] = frame_detections["frame_id"]
            if individual_id in individual_tracks.keys():
                if segment_id in individual_tracks[individual_id].keys():
                    individual_tracks[individual_id][segment_id].append(detection)
                else:
                    individual_tracks[individual_id][segment_id] = [detection]
            else:
                individual_tracks[individual_id] = {segment_id: [detection]}

    # Condense information
    for i, it in individual_tracks.items():
        individual_tracks[i] = [s for s in it.values()]
    individual_tracks = [it for it in individual_tracks.values()]

    return individual_tracks


### Parsing functions
def get_tracks_from_attribute(individual_tracks: List[IndividualTrack], attribute_name: str, attribute_value: str):
    """
    Retrieve tracks corresponding to a given attribute name and value.

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
            for frame in segment:
                if "attributes" in frame and frame["attributes"].get(attribute_name) == attribute_value:
                    attr_tracks.append(track)
                    found = True
                    break
            if found:
                break  # Stop processing this track after first match

    return attr_tracks

def get_tracks_from_species(individual_tracks: List[IndividualTrack], species_name: str
) -> List[IndividualTrack]:
    """
    Retrieve tracks corresponding to a given species name
    """
    return get_tracks_from_attribute(individual_tracks, "Species", species_name)


def get_tracks_from_action(
    individual_tracks: List[IndividualTrack], action_name: str
) -> List[IndividualTrack]:
    """
    Retrieve tracks corresponding to a given action
    """
    return get_tracks_from_attribute(individual_tracks, "Action", action_name) + get_tracks_from_attribute(individual_tracks, "Action2", action_name)


def get_tracks_from_activity(
    individual_tracks: List[IndividualTrack], activity_name: str
) -> List[IndividualTrack]:
    """
    Retrieve tracks corresponding to a given activity
    """
    
    return get_tracks_from_attribute(individual_tracks, "Activity", activity_name)


def get_deer_tracks_from_age(individual_tracks: List[IndividualTrack], age: str) -> List[IndividualTrack]:
    """
    Retrieve deer tracks corresponding to a given age
    """

    deer_tracks = (get_tracks_from_species(individual_tracks, "red_deer") + get_tracks_from_species(individual_tracks, "roe_deer"))
    return get_tracks_from_attribute(deer_tracks, "Deer_age", age)


def get_adult_deer_tracks_from_sex(
    individual_tracks: List[IndividualTrack], sex: str
) -> List[IndividualTrack]:
    """
    Retrieve adult deer tracks corresponding to a given sex
    """

    adult_deer_tracks = get_deer_tracks_from_age(individual_tracks, "adult")
    return get_tracks_from_attribute(adult_deer_tracks, "Deer_adult_sex", sex)


def get_nb_tracks_species_in_video(individual_tracks: List[IndividualTrack], species_name) -> int:
    """
    Returns the number of tracks for a given species in the video
    """

    return len(get_tracks_from_species(individual_tracks, species_name))


def video_contains_species(individual_tracks: List[IndividualTrack], species_name: str, min_occurences: int = 1) -> bool:
    """
    Returns True if the video contains tracks with the given species
    """

    return get_nb_tracks_species_in_video(individual_tracks, species_name) >= min_occurences


def get_nb_tracks_action_in_video(individual_tracks: List[IndividualTrack], action_name: str) -> int:
    """
    Returns the number of tracks for a given action in the video
    """
    return len(get_tracks_from_action(individual_tracks, action_name))


def video_contains_action(individual_tracks: List[IndividualTrack], action_name: str, min_occurences: int = 1) -> bool:
    """
    Returns True if the video contains tracks with the given action
    """
    return get_nb_tracks_action_in_video(individual_tracks, action_name) >= min_occurences


def get_nb_tracks_activity_in_video(individual_tracks: List[IndividualTrack], activity_name: str) -> int:
    """
    Returns the number of tracks for a given activity in the video
    """
    return len(get_tracks_from_activity(individual_tracks, activity_name))


def video_contains_activity(individual_tracks: List[IndividualTrack], activity_name: str, min_occurences: int = 1) -> bool:
    """
    Returns True if the video contains tracks with the given activity
    """
    return get_nb_tracks_activity_in_video(individual_tracks, activity_name) >= min_occurences


def get_nb_deer_tracks_age_in_video(individual_tracks: List[IndividualTrack], age: str) -> int:
    """
    Returns the number of deer tracks for a given age in the video
    """
    return len(get_deer_tracks_from_age(individual_tracks, age))


def video_contains_deer_age(individual_tracks: List[IndividualTrack], age: str, min_occurences: int = 1) -> bool:
    """
    Returns True if the video contains deer tracks with the given age
    """
    return get_nb_deer_tracks_age_in_video(individual_tracks, age) >= min_occurences


def get_nb_adult_deer_tracks_sex_in_video(individual_tracks: List[IndividualTrack], sex: str) -> int:
    """
    Returns the number of adult deer tracks for a given sex in the video
    """
    return len(get_adult_deer_tracks_from_sex(individual_tracks, sex))


def video_contains_adult_deer_sex(individual_tracks: List[IndividualTrack], sex: str, min_occurences: int = 1) -> bool:
    """
    Returns True if the video contains adult deer tracks with the given sex
    """
    return get_nb_adult_deer_tracks_sex_in_video(individual_tracks, sex) >= min_occurences