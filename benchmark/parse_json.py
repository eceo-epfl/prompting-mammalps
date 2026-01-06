import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

FrameDetections = Dict
VideoDict = List[FrameDetections]
InfoDict = Dict

BehaviorSegment = List[Dict]
IndividualTrack = List[BehaviorSegment]

Point = Tuple[int]
BBox = Tuple[Point]
Mask = List[Point]


### Basic functions
def load_video_detections(json_file: Union[Path, str]) -> VideoDict:
    with open(json_file, "r") as f:
        content = json.load(f)

    return content["frames"]


def load_video_info(json_file: Union[Path, str]) -> InfoDict:
    with open(json_file, "r") as f:
        content = json.load(f)

    return content["info"]


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
    segment: BehaviorSegment, attribute_name: str, attribute_value: str
):
    # We can check only the first element of the segment since they all contain the same attributes
    return (
        "attributes" in segment[0]
        and segment[0]["attributes"].get(attribute_name) == attribute_value
    )


def get_tracks_from_attribute(
    individual_tracks: List[IndividualTrack], attribute_name: str, attribute_value: str
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
    individual_tracks: List[IndividualTrack], attribute_name: str, attribute_value: str
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
    individual_tracks: List[IndividualTrack], attribute_name: str, attribute_value: str
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
        if len(track):
            attr_tracks.append(track)

    return attr_tracks


def get_tracks_from_species(
    individual_tracks: List[IndividualTrack], species_name: str
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
    attr_tracks = get_segments_from_attribute_as_tracks(
        individual_tracks, "Action", action_name
    ) + get_segments_from_attribute_as_tracks(individual_tracks, "Action2", action_name)

    return attr_tracks  # duplicates if action = action2


def get_tracks_from_activity(
    individual_tracks: List[IndividualTrack], activity_name: str
) -> List[IndividualTrack]:
    """
    Retrieve tracks corresponding to a given activity
    """

    return get_segments_from_attribute_as_tracks(
        individual_tracks, "Activity", activity_name
    )


def get_deer_tracks_from_age(
    individual_tracks: List[IndividualTrack],
    age: str,
    deer_species: Optional[str] = None,
) -> List[IndividualTrack]:
    """
    Retrieve deer tracks corresponding to a given age
    """
    if deer_species is not None:
        deer_tracks = get_tracks_from_species(
            individual_tracks, species_name=deer_species
        )
    else:
        deer_tracks = get_tracks_from_species(
            individual_tracks, "red_deer"
        ) + get_tracks_from_species(individual_tracks, "roe_deer")
    return get_tracks_from_attribute(deer_tracks, "Deer_age", age)


def get_adult_deer_tracks_from_sex(
    individual_tracks: List[IndividualTrack],
    sex: str,
    deer_species: Optional[str] = None,
) -> List[IndividualTrack]:
    """
    Retrieve adult deer tracks corresponding to a given sex
    """

    adult_deer_tracks = get_deer_tracks_from_age(
        individual_tracks, "adult", deer_species=deer_species
    )
    return get_tracks_from_attribute(adult_deer_tracks, "Deer_adult_sex", sex)


def get_nb_tracks_species_in_video(
    individual_tracks: List[IndividualTrack], species_name
) -> int:
    """
    Returns the number of tracks for a given species in the video
    """

    return len(get_tracks_from_species(individual_tracks, species_name))


def tracks_contain_species(
    individual_tracks: List[IndividualTrack], species_name: str
) -> bool:
    """
    Returns True if the video contains tracks with the given species
    """

    return get_nb_tracks_species_in_video(individual_tracks, species_name) >= 1


def get_nb_tracks_action_in_video(
    individual_tracks: List[IndividualTrack], action_name: str
) -> int:
    """
    Returns the number of tracks for a given action in the video
    """
    return len(get_tracks_from_action(individual_tracks, action_name))


def tracks_contain_action(
    individual_tracks: List[IndividualTrack], action_name: str
) -> bool:
    """
    Returns True if the video contains tracks with the given action
    """
    return get_nb_tracks_action_in_video(individual_tracks, action_name) >= 1


def get_nb_tracks_activity_in_video(
    individual_tracks: List[IndividualTrack], activity_name: str
) -> int:
    """
    Returns the number of tracks for a given activity in the video
    """
    return len(get_tracks_from_activity(individual_tracks, activity_name))


def tracks_contain_activity(
    individual_tracks: List[IndividualTrack],
    activity_name: str,
) -> bool:
    """
    Returns True if the video contains tracks with the given activity
    """
    return get_nb_tracks_activity_in_video(individual_tracks, activity_name) >= 1


def get_nb_deer_tracks_age_in_video(
    individual_tracks: List[IndividualTrack],
    age: str,
    deer_species: Optional[str] = None,
) -> int:
    """
    Returns the number of deer tracks for a given age in the video
    """
    return len(
        get_deer_tracks_from_age(individual_tracks, age, deer_species=deer_species)
    )


def tracks_contain_deer_age(
    individual_tracks: List[IndividualTrack],
    age: str,
    deer_species: Optional[str] = None,
) -> bool:
    """
    Returns True if the video contains deer tracks with the given age
    """
    return (
        get_nb_deer_tracks_age_in_video(
            individual_tracks, age, deer_species=deer_species
        )
        >= 1
    )


def get_nb_adult_deer_tracks_sex_in_video(
    individual_tracks: List[IndividualTrack],
    sex: str,
    deer_species: Optional[str] = None,
) -> int:
    """
    Returns the number of adult deer tracks for a given sex in the video
    """
    return len(
        get_adult_deer_tracks_from_sex(
            individual_tracks, sex, deer_species=deer_species
        )
    )


def tracks_contain_adult_deer_sex(
    individual_tracks: List[IndividualTrack],
    sex: str,
    deer_species: Optional[str] = None,
) -> bool:
    """
    Returns True if the video contains adult deer tracks with the given sex
    """
    return (
        get_nb_adult_deer_tracks_sex_in_video(
            individual_tracks, sex, deer_species=deer_species
        )
        >= 1
    )


## Attributes and location
def get_tracks_overlapping_point(
    individual_tracks: List[IndividualTrack], point: Point
) -> List[IndividualTrack]:
    """
    Retrieve tracks that overlap with a given point
    Args:
        individual_tracks: Dictionary containing video detection data
        point: Point coordinates as (x,y) tuple
    Returns:
        List of tracks that overlap with the point
    """
    pass


def get_tracks_overlapping_bbox(
    individual_tracks: List[IndividualTrack], bbox: BBox
) -> List[IndividualTrack]:
    """
    Retrieve tracks that overlap with a given bounding box
    Args:
        individual_tracks: Dictionary containing video detection data
        bbox: Bounding box coordinates as ((xtl,ytl),(xbr,ybr)) tuple
    Returns:
        List of tracks that overlap with the bounding box
    """
    pass


def get_tracks_overlapping_mask(
    individual_tracks: List[IndividualTrack], mask: Mask
) -> List[IndividualTrack]:
    """
    Retrieve tracks that overlap with a given mask
    Args:
        individual_tracks: Dictionary containing video detection data
        mask: List of (x,y) points defining a polygon mask
    Returns:
        List of tracks that overlap with the mask
    """
    pass


## Attributes and video
def tracks_contain_same_activities_as_ref(
    individual_tracks: List[IndividualTrack], ref_json: VideoDict
) -> bool:
    pass


def tracks_contain_same_species_as_ref():
    pass


def tracks_contain_same_activity_sequence_as_ref():
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parse video detection JSON file.")
    parser.add_argument("--input_json", type=str, help="Path to the input JSON file")
    args = parser.parse_args()

    video_detections = load_video_detections(args.input_json)
    video_info = load_video_info(args.input_json)

    tracks = get_tracks_from_video_detections(video_detections)
