"""
Splits the videos into train, test and validation splits.
Events are considered dependent if they occur on the same day, so events in the val and test splits occur on different days than on the train split
"""

import argparse
import json

import numpy as np
import pandas as pd

SEED = 0

DISCARD = ["E460"]  # Single falling activity

MANDATORY_TRAIN = [
    "E23",
    "E55",
    "E65",
    "E75",
    "E131",
    "E163",
    "E200",
    "E204",
    "E243",
    "E386",
    "E594",
    "E474",
    "E475",
    "E548",
    "E700",
    "E746",
    "E737",
]  # For rare actions/activities and for rare species
MANDATORY_TEST = [
    "E168",
    "E169",
    "E230",
    "E498",
    "E491",
    "E555",
    "E568",
    "E573",
    "E464",
    "E720",
    "E863",
    "E874",
]  # For rare actions/activities and for rare species


def perform_split(options):
    # TODO: Here it drops when there are more than one segment for an individual
    annotations_df = pd.read_csv(options.annotations_csv, index_col=0).drop_duplicates(
        ["file_id", "tublet_file_id"]
    )
    annotations_df["date_time"] = pd.to_datetime(annotations_df["date_time"])

    # Remove undesired events
    annotations_df = annotations_df[
        annotations_df["event_id"].isin(DISCARD) == False
    ].reset_index(drop=True)

    if options.split_type == "RANDOM":
        train_events, val_events, test_events = random_day_split(
            annotations_df[
                ["event_id", "tublet_file_id", "date_time", "tublet_duration"]
            ],
            train_size=options.train_size,
            test_size=options.test_size,
            train_init=MANDATORY_TRAIN,
            test_init=MANDATORY_TEST,
            seed=SEED,
        )

        annotations_df["subset"] = annotations_df["event_id"].map(
            {
                **(dict(zip(train_events, ["train"] * len(train_events)))),
                **(dict(zip(val_events, ["validation"] * len(val_events)))),
                **(dict(zip(test_events, ["test"] * len(test_events)))),
            }
        )

        events_split = {
            "train": train_events.tolist(),
            "val": val_events.tolist(),
            "test": test_events.tolist(),
        }

        with open(opt.output_json, "w") as f:
            json.dump(events_split, f, indent=2)


def random_day_split(
    annotations_df, train_size, test_size, train_init=[], test_init=[], seed=0
):
    """
    Splits events so that they train/val/test events belong to different random days.
    """
    train_init_dates = annotations_df[annotations_df["event_id"].isin(train_init)][
        "date_time"
    ].dt.date.unique()
    test_init_dates = annotations_df[annotations_df["event_id"].isin(test_init)][
        "date_time"
    ].dt.date.unique()
    assert (
        len(set(train_init_dates).intersection(set(test_init_dates))) == 0
    ), "some days in the provided initial train set overlap with the initial test set"

    tublets_per_day = (
        annotations_df[["date_time", "tublet_duration"]]
        .groupby([annotations_df["date_time"].dt.date])[["tublet_duration"]]
        .sum()
        .reset_index()
    )
    init_train_prop = (
        tublets_per_day[tublets_per_day["date_time"].isin(train_init_dates)][
            "tublet_duration"
        ].sum()
        / tublets_per_day["tublet_duration"].sum()
    )
    init_test_prop = (
        tublets_per_day[tublets_per_day["date_time"].isin(test_init_dates)][
            "tublet_duration"
        ].sum()
        / tublets_per_day["tublet_duration"].sum()
    )

    print(tublets_per_day[tublets_per_day["date_time"].isin(test_init_dates)])
    print(tublets_per_day[tublets_per_day["date_time"].isin(train_init_dates)])

    print("Init train set prop:", init_train_prop)
    print("Init test set prop:", init_test_prop)

    success = False

    while not success:
        try:
            # We remove predefined days and shuffle the rest
            tublets_per_day_without_init = tublets_per_day[
                ~(
                    tublets_per_day["date_time"].isin(
                        np.concatenate([train_init_dates, test_init_dates])
                    )
                )
            ]
            tublets_per_day_without_init = tublets_per_day_without_init.sample(
                n=len(tublets_per_day_without_init), random_state=seed
            )

            # We append the train days at the beggining, and the test days at the end.
            tublets_per_day_with_init = pd.concat(
                (
                    tublets_per_day[
                        tublets_per_day["date_time"].isin(train_init_dates)
                    ],
                    tublets_per_day_without_init,
                    tublets_per_day[tublets_per_day["date_time"].isin(test_init_dates)],
                ),
                axis=0,
            )

            # We compute cumsum duration of tublets per day and perform split based on provided times
            tublets_per_day_with_init["cum_duration"] = tublets_per_day_with_init[
                "tublet_duration"
            ].cumsum()

            tot_duration = tublets_per_day_with_init["tublet_duration"].sum()
            train_days = tublets_per_day_with_init[
                tublets_per_day_with_init["cum_duration"] <= train_size * tot_duration
            ]["date_time"]

            val_days = tublets_per_day_with_init[
                (tublets_per_day_with_init["cum_duration"] > train_size * tot_duration)
                & (
                    tublets_per_day_with_init["cum_duration"]
                    <= min(
                        (1 - test_size) * tot_duration,
                        (1 - init_test_prop) * tot_duration,
                    )
                )
            ]["date_time"]

            test_days = tublets_per_day_with_init.drop(train_days.index).drop(
                val_days.index
            )["date_time"]

            train_df = annotations_df[
                annotations_df["date_time"].dt.date.isin(train_days)
            ]
            val_df = annotations_df[annotations_df["date_time"].dt.date.isin(val_days)]
            test_df = annotations_df[
                annotations_df["date_time"].dt.date.isin(test_days)
            ]

            train_events = train_df.event_id.unique()
            val_events = val_df.event_id.unique()
            test_events = test_df.event_id.unique()

            # train_prop = tublets_per_day[tublets_per_day["date_time"].isin(train_days)]["tublet_duration"].sum()/tublets_per_day["tublet_duration"].sum()
            test_prop = (
                tublets_per_day[tublets_per_day["date_time"].isin(test_days)][
                    "tublet_duration"
                ].sum()
                / tublets_per_day["tublet_duration"].sum()
            )

            # May occur for events around 12.00am
            assert len(set(train_events).intersection(set(val_events))) == 0
            assert len(set(train_events).intersection(set(test_events))) == 0
            assert len(set(test_events).intersection(set(val_events))) == 0

            assert test_prop < (test_size + 0.02)

            assert len(test_df[test_df["event_id"].isin(train_init)]) == 0
            assert len(val_df[val_df["event_id"].isin(train_init)]) == 0
            assert len(train_df[train_df["event_id"].isin(test_init)]) == 0
            assert len(val_df[val_df["event_id"].isin(test_init)]) == 0

            assert (
                len(
                    set(train_df["date_time"].dt.date).intersection(
                        set(val_df["date_time"].dt.date)
                    )
                )
                == 0
            )
            assert (
                len(
                    set(train_df["date_time"].dt.date).intersection(
                        set(test_df["date_time"].dt.date)
                    )
                )
                == 0
            )
            assert (
                len(
                    set(test_df["date_time"].dt.date).intersection(
                        set(val_df["date_time"].dt.date)
                    )
                )
                == 0
            )

            success = True

        except AssertionError as e:
            success = False
            seed += 1

    print(
        "Train prop:",
        train_df["tublet_duration"].sum()
        / (
            train_df["tublet_duration"].sum()
            + test_df["tublet_duration"].sum()
            + val_df["tublet_duration"].sum()
        ),
    )
    print(
        "Val prop:",
        val_df["tublet_duration"].sum()
        / (
            train_df["tublet_duration"].sum()
            + test_df["tublet_duration"].sum()
            + val_df["tublet_duration"].sum()
        ),
    )
    print(
        "Test prop:",
        test_df["tublet_duration"].sum()
        / (
            train_df["tublet_duration"].sum()
            + test_df["tublet_duration"].sum()
            + val_df["tublet_duration"].sum()
        ),
    )

    return train_events, val_events, test_events


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-I",
        "--annotations_csv",
        help="CSV file containing the fields event_id, date_time, file_id, tublet_duration, tublet_file_id",
    )
    parser.add_argument("--train_size", type=float, help="proportion of train set")
    parser.add_argument(
        "--test_size",
        type=float,
        help="proportion of test set, remaining is the val set",
    )
    parser.add_argument(
        "--split_type", type=str, choices=["RANDOM"], help="type of train/test split"
    )
    parser.add_argument(
        "-O",
        "--output_json",
        help="Output JSON dict containing the list of events per split",
    )

    opt = parser.parse_args()

    perform_split(opt)
