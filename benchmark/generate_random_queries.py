import argparse
import random
from enum import Enum


class Species(Enum):
    RED_DEER = "red deer"
    ROE_DEER = "roe deer"
    FOX = "fox"
    HARE = "mountain hare"
    MARTEN = "marten"
    WOLF = "wolf"
    CHAMOIS = "chamois"


species = [s.value for s in Species]
rare_species = [s.value for s in Species if s != Species.RED_DEER]


class Action(Enum):
    WALKING = "walking"
    STANDING_HEAD_UP = "standing head up"
    STANDING_HEAD_DOWN = "standing head down"
    GRAZING = "grazing"
    SNIFFING = "sniffing"
    LOOKING_AT_CAMERA = "looking at a camera"
    TROTTING_OR_RUNNING = "trotting or running"
    SCRATCHING_OWN_HEAD_OR_BODY = "scratching its own head or body"
    RUBBING_ANTLERS_ON_GROUND = "rubbing its antlers on the ground"
    PAWING_GROUND = "pawing the ground"
    SHAKING_HEAD_OR_BODY = "shaking its head or body"
    VOCALIZING = "vocalizing"
    BATHING = "bathing"
    JUMPING = "jumping"
    DRINKING = "drinking"
    LAYING_DOWN = "laying down"
    DEFECATING = "defecating"
    URINATING = "urinating"
    BROWSING = "browsing"
    STRETCHING_BODY = "stretching its body"
    SUCKLING = "suckling"
    PREPARING_TO_SUCKLE = "preparing to suckle"


actions = [a.value for a in Action]


class Activity(Enum):
    FORAGING = "foraging"
    VIGILANCE = "in vigilance"
    COURTSHIP = "participating in courtship"
    CAMERA_REACTION = "reacting to the camera"
    ESCAPING = "escaping"
    CHASING = "chasing"
    NURSING = "nursing"
    GROOMING = "grooming itself"
    PLAYING = "playing"
    RESTING = "resting"
    MARKING_OR_WALLOWING = "marking or wallowing"


activities = [a.value for a in Activity]
rare_activities = [
    a.value for a in Activity if a not in [Activity.FORAGING, Activity.VIGILANCE]
]
social_activities = [
    a.value
    for a in Activity
    if a
    in [
        Activity.VIGILANCE,
        Activity.FORAGING,
        Activity.NURSING,
    ]
]

courtship_actn = [
    Action.WALKING.value,
    Action.STANDING_HEAD_UP.value,
    Action.STANDING_HEAD_DOWN.value,
    Action.SNIFFING.value,
    Action.SCRATCHING_OWN_HEAD_OR_BODY.value,
    Action.RUBBING_ANTLERS_ON_GROUND.value,
    Action.PAWING_GROUND.value,
    Action.SHAKING_HEAD_OR_BODY.value,
    Action.VOCALIZING.value,
    Action.BATHING.value,
    Action.JUMPING.value,
    Action.LAYING_DOWN.value,
    Action.URINATING.value,
]

rare_actions = [
    a.value
    for a in Action
    if a
    not in [
        Action.WALKING,
        Action.STANDING_HEAD_UP,
        Action.STANDING_HEAD_DOWN,
        Action.GRAZING,
    ]
    + courtship_actn
]

social_actions = [
    a.value
    for a in Action
    if a
    in [
        Action.WALKING,
        Action.STANDING_HEAD_UP,
    ]
]


class DAge(Enum):
    """Deer age"""

    ADULT = "adult"
    JUVENILE = "juvenile"


deer_ages = [a.value for a in DAge]


class DSex(Enum):
    "Sex for adult deers"
    MALE = "male"
    FEMALE = "female"


deer_adult_sexes = [a.value for a in DSex]


class Meteo(Enum):
    SUNNY = "sunny"
    CLEAR = "clear"
    OVERCAST = "overcast"
    RAINY = "rainy"


meteo = [m.value for m in Meteo]


class NbInd(Enum):
    TWO = "two"
    THREE_OR_MORE = "three or more"


nb_ind = [n.value for n in NbInd]


def generate_queries(opt):

    queries = []

    templates = []
    with open(opt.templates, "r") as f:
        content = f.read().splitlines()
        for line in content:
            if line and line[0] != "#":
                templates.append(line)

    templates = [t for t in templates if (("[Loc]" not in t) and ("[Vid]" not in t))]

    for i in range(opt.n_queries_base):
        query = random.choice(templates)
        if "[ActY]" in query:
            acty = random.choice(activities)
            query = query.replace("[ActY]", acty)
        if "[ActN]" in query:
            actn = random.choice(actions)
            query = query.replace("[ActN]", actn)
        if "[Spe]" in query:
            spe = random.choice(species)
            query = query.replace("[Spe]", spe)
        if "[NbInd]" in query:
            nb = random.choice(nb_ind)
            query = query.replace("[NbInd]", nb)
        if "[DAge]" in query:
            deer_age = random.choice(deer_ages)
            query = query.replace("[DAge]", deer_age)
        if "[DSex]" in query:
            deer_sex = random.choice(deer_adult_sexes)
            query = query.replace("[DSex]", deer_sex)

        queries.append(query)

    return queries


def categorize_queries(queries: list[str], opt) -> dict[str, list]:
    queries_per_cat = {}

    # RARE VIDEOS
    rare_queries = [
        q
        for q in queries
        if (
            (
                any(x in q for x in rare_species)
                or any(x in q for x in rare_activities)
                or any(x in q for x in rare_actions)
            )
            and not any(x in q for x in [NbInd.TWO.value, NbInd.THREE_OR_MORE.value])
        )
    ]
    queries_per_cat["rare"] = random.sample(
        rare_queries, k=min(opt.n_rare, len(rare_queries))
    )

    # COURTSHIP
    courtship_queries = [
        q
        for q in queries
        if (
            any(x in q for x in courtship_actn)
            and any(x in q for x in [Species.RED_DEER.value, Species.ROE_DEER.value])
            and Activity.COURTSHIP.value in q
            and not DAge.JUVENILE.value in q
        )
    ]
    queries_per_cat["courtship"] = random.sample(
        courtship_queries, k=min(opt.n_courtship, len(courtship_queries))
    )

    # SOCIAL
    social_queries = [
        q
        for q in queries
        if (
            any(x in q for x in social_actions)
            and any(x in q for x in social_activities)
            and Species.RED_DEER.value in q
            and not Activity.COURTSHIP.value in q
        )
    ]
    queries_per_cat["social_deer"] = random.sample(
        social_queries, k=min(opt.n_social, len(social_queries))
    )

    # CAM REACTION
    cam_reaction_queries = [q for q in queries if (Activity.CAMERA_REACTION.value in q)]
    queries_per_cat["cam_reaction"] = random.sample(
        cam_reaction_queries, k=min(opt.n_cam_reaction, len(cam_reaction_queries))
    )

    return queries_per_cat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-I", "--templates", type=str, help="File containing queries templates"
    )
    parser.add_argument(
        "-N",
        "--n_queries_base",
        type=int,
        help="Hidden number of queries before categorization",
        default=50000,
    )
    parser.add_argument(
        "--n_rare", type=int, help="Number of queries for the rare event category"
    )
    parser.add_argument(
        "--n_courtship", type=int, help="Number of queries for the courtship category"
    )
    parser.add_argument(
        "--n_social", type=int, help="Number of events for the social deer category"
    )
    parser.add_argument(
        "--n_cam_reaction",
        type=int,
        help="Number of reaction to camera queries category",
    )

    opt = parser.parse_args()

    queries = generate_queries(opt)

    queries_per_cat = categorize_queries(queries, opt)

    print("[RARE]", *queries_per_cat["rare"], sep="\n")

    print("[COURTSHIP]", *queries_per_cat["courtship"], sep="\n")

    print("[SOCIAL]", *queries_per_cat["social_deer"], sep="\n")

    print("[CAM]", *queries_per_cat["cam_reaction"], sep="\n")
