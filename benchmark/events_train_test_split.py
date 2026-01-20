"""
Splits the videos into train, test and validation splits.
Events are considered dependent if they occur on the same day, so events in the val and test splits occur on different days than on the train split
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


def random_day_split(
    annotations_df,
    train_size,
    test_size,
    train_init=[],
    test_init=[],
    seed=0,
    no_val=False,
    n_iter=10000,
):
    """
    Splits events so that they train/val/test events belong to different random days.
    """
    # Check size arguments
    if no_val:
        val_size = 0
        assert (
            train_size + test_size == 1.0
        ), "No validation set, train size and test size must sum to 1."
    else:
        val_size = 1 - train_size - test_size
        assert (
            val_size >= 0
        ), "With a validation set, remaining validation set size must be >= 0"
        assert (
            val_size < 1
        ), "With a validation set, remaining validation set size must be < 1"

    train_init_dates = annotations_df[annotations_df["event_id"].isin(train_init)][
        "date_time"
    ].dt.date.unique()
    test_init_dates = annotations_df[annotations_df["event_id"].isin(test_init)][
        "date_time"
    ].dt.date.unique()
    overlapping_days = set(train_init_dates).intersection(set(test_init_dates))
    assert (
        len(overlapping_days) == 0
    ), f"some days in the provided initial train set overlap with the initial test set: {overlapping_days}"

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

    if val_size > 1 - init_train_prop - init_test_prop:
        new_val_size = max(1 - init_train_prop - init_test_prop, 0)
        print(
            f"Given the initial set proportions, decreasing validation set size from {val_size} to {new_val_size}"
        )
        val_size = new_val_size

    orig_train_size = train_size
    if test_size < init_test_prop:
        new_train_size = 1 - init_test_prop - 0.005
        print(
            f"Given the initial set proportions, decreasing train set size from {train_size} to {new_train_size}"
        )
        train_size = new_train_size

    print("Search parameters:")
    print(
        "\t Target train size:",
        np.round(train_size, 3),
        "- Target val size:",
        np.round(val_size, 3),
        "- Target test size:",
        np.round(test_size, 3),
    )
    print("\t Init train set prop:", np.round(init_train_prop, 3))
    print("\t Init test set prop:", np.round(init_test_prop, 3))
    print("\t Number of iterations:", n_iter)

    best_split = ()
    best_split_score = 1
    tot_duration = tublets_per_day["tublet_duration"].sum()
    init_dates = np.concatenate([train_init_dates, test_init_dates])

    for i in tqdm(range(n_iter)):
        try:
            # We remove predefined days and shuffle the rest
            tublets_per_day_without_init = tublets_per_day[
                ~(tublets_per_day["date_time"].isin(init_dates))
            ]
            tublets_per_day_without_init = tublets_per_day_without_init.sample(
                n=len(tublets_per_day_without_init), random_state=i
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

            # train days: until cumsum of tublet_duration reaches train_size * tot_duration
            train_days = tublets_per_day_with_init[
                tublets_per_day_with_init["cum_duration"] <= train_size * tot_duration
            ]["date_time"]

            # val days: until cumsum of tublet_duration reaches train_size + val_size * tot_duration
            val_days = tublets_per_day_with_init[
                (tublets_per_day_with_init["cum_duration"] > train_size * tot_duration)
                & (
                    tublets_per_day_with_init["cum_duration"]
                    <= (train_size + val_size) * tot_duration
                )
            ]["date_time"]

            # test days: remaining days
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

            train_prop = (
                tublets_per_day[tublets_per_day["date_time"].isin(train_days)][
                    "tublet_duration"
                ].sum()
                / tot_duration
            )
            val_prop = (
                tublets_per_day[tublets_per_day["date_time"].isin(val_days)][
                    "tublet_duration"
                ].sum()
                / tot_duration
            )
            test_prop = (
                tublets_per_day[tublets_per_day["date_time"].isin(test_days)][
                    "tublet_duration"
                ].sum()
                / tot_duration
            )

            # Check sets don't overlap
            # May occur for events around 12.00am
            assert len(set(train_events).intersection(set(val_events))) == 0
            assert len(set(train_events).intersection(set(test_events))) == 0
            assert len(set(test_events).intersection(set(val_events))) == 0

            # Check init sets are part of the final sets
            assert len(set(train_events).intersection(set(train_init))) == len(
                set(train_init)
            )
            assert len(set(test_events).intersection(set(test_init))) == len(
                set(test_init)
            )

            # Check dates don't overlap
            assert len(set(train_days).intersection(set(val_days))) == 0
            assert len(set(train_days).intersection(set(test_days))) == 0
            assert len(set(val_days).intersection(set(test_days))) == 0

            # No error, this is a valid set, compute split score
            split_score = (
                abs(train_prop - orig_train_size)
                + abs(test_prop - test_size)
                + abs(val_prop - val_size)
            )
            if split_score < best_split_score:
                best_split_score = split_score
                best_split = (train_events, val_events, test_events)

        except AssertionError as e:
            pass

    if len(best_split) == 0:
        print("No combination found given the initial train/test days and set sizes.")
        return [], [], []
    else:
        train_events, val_events, test_events = best_split

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
