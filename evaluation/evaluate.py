#!/usr/bin/env python3
"""Evaluation script for the retrieval pipeline in the paper.

`retrieval` mode evaluates a set of retrieved video ids per query.

Expects `--data-root` to point at a local copy of the
amathislab/Prompting-MammAlps HF dataset (see README for download
instructions) so that `metadata/query_categories.json` and the video-id
universe (`annotations/`) can be found.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def F1_score_q(gt_videos, pred_videos) -> float:
    gt_videos = set(gt_videos)
    pred_videos = set(pred_videos)

    tp = len(pred_videos & gt_videos)
    fp = len(pred_videos - gt_videos)
    fn = len(gt_videos - pred_videos)

    if 2 * tp + fp + fn == 0:
        f1_score = np.nan
    else:
        f1_score = 2 * tp / (2 * tp + fp + fn)

    return f1_score


def F1_score(gt_queries_videos, pred_queries_videos, all_videos) -> tuple:

    F1i = {}

    for q in gt_queries_videos.keys():
        if q not in pred_queries_videos:
            F1i[q] = 0
            continue
        gt_videos_q = list(gt_queries_videos[q])
        ass_videos_q = list(pred_queries_videos[q])

        if len(gt_videos_q) == 0:  # Empty query: we measure the opposite
            gt_videos_q = all_videos
            ass_videos_q = list(set(all_videos) - set(ass_videos_q))

        F1i[q] = F1_score_q(gt_videos_q, ass_videos_q)

    return np.nanmean(list(F1i.values())), F1i


def load_query_categories(data_root: Path) -> dict:
    with open(Path(data_root) / "metadata" / "query_categories.json", "r") as f:
        return json.load(f)


def load_video_universe(data_root: Path) -> tuple[set[str], set[str]]:
    """Returns (test_video_ids, train_video_ids) from the local annotation files.

    Retrieval is only ever evaluated over the test set, so callers must not mix
    the two universes when scoring a predictions file.
    """
    test_video_ids = {
        f.stem for f in (data_root / "annotations" / "test").rglob("*.json")
    }
    train_video_ids = {
        f.stem for f in (data_root / "annotations" / "train").rglob("*.json")
    }
    return test_video_ids, train_video_ids


def validate_predictions(
    pred_videos: dict, gt_queries, test_video_ids: set[str], train_video_ids: set[str]
) -> list[str]:
    """Sanity-checks a predictions dict before scoring it.

    Returns a list of human-readable issues; empty if the file is well-formed.
    """
    errors = []

    gt_queries = set(gt_queries)
    pred_queries = set(pred_videos)

    missing = gt_queries - pred_queries
    if missing:
        errors.append(
            f"{len(missing)} benchmark quer{'y is' if len(missing) == 1 else 'ies are'} "
            f"missing from predictions: {sorted(missing)}"
        )

    unknown_queries = pred_queries - gt_queries
    if unknown_queries:
        errors.append(
            f"{len(unknown_queries)} predicted quer{'y does' if len(unknown_queries) == 1 else 'ies do'} "
            f"not match any benchmark query (typo?): {sorted(unknown_queries)}"
        )

    for q, ids in pred_videos.items():
        if not isinstance(ids, list) or not all(isinstance(v, str) for v in ids):
            errors.append(
                f"Query {q!r}: predicted value must be a list of video id strings."
            )
            continue
        dupes = sorted({v for v in ids if ids.count(v) > 1})
        if dupes:
            errors.append(f"Query {q!r}: duplicate video id(s) in prediction: {dupes}")

    predicted_ids = {
        v for ids in pred_videos.values() if isinstance(ids, list) for v in ids
    }

    train_leak = predicted_ids & train_video_ids
    if train_leak:
        errors.append(
            f"{len(train_leak)} predicted video id(s) belong to the train set, not test: {sorted(train_leak)}"
        )

    unknown_ids = predicted_ids - test_video_ids - train_video_ids
    if unknown_ids:
        errors.append(
            f"{len(unknown_ids)} predicted video id(s) do not exist in the dataset: {sorted(unknown_ids)}"
        )

    return errors


def build_category_maps(query_categories: dict, queries) -> tuple[dict, dict]:
    """Maps each query to its ecology and computer-vision complexity categories."""
    query2eco_cat = {q: [] for q in queries}
    for cat, cat_queries in query_categories.get("ECOLOGY", {}).items():
        for q in cat_queries:
            if q in query2eco_cat:
                query2eco_cat[q].append(cat)

    query2cv_cat = {q: [] for q in queries}
    for cat, cat_queries in query_categories.get("VISION", {}).items():
        for q in cat_queries:
            if q in query2cv_cat:
                query2cv_cat[q].append(cat)

    return query2eco_cat, query2cv_cat


def category_breakdown(scores: dict, query2cat: dict, label: str) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(scores, orient="index", columns=[label])
    cat_df = pd.DataFrame([query2cat], index=["category"]).T
    return (
        df.merge(cat_df, left_index=True, right_index=True)
        .explode("category")
        .groupby("category")
        .mean()
        .round(2)
    )


def report(
    scores_by_metric: dict, query2eco_cat: dict, query2cv_cat: dict, output: str | None
):
    results_df = pd.concat(
        [
            pd.DataFrame.from_dict(scores, orient="index", columns=[name])
            for name, scores in scores_by_metric.items()
        ],
        axis=1,
    )
    print("\nPer-query results:")
    print(
        results_df.sort_values(results_df.columns[0], ascending=False)
        .round(2)
        .to_string()
    )

    for name, scores in scores_by_metric.items():
        for cat_label, query2cat in [
            ("ecology", query2eco_cat),
            ("vision", query2cv_cat),
        ]:
            print(f"\n{name} by {cat_label} category:")
            print(category_breakdown(scores, query2cat, name).to_string())

    print("\nOverall results:")
    print(
        "\tF1-score (macro-avg.): "
        f"{np.nanmean([s for _, s in scores_by_metric['F1-score'].items()]):.2f}"
    )
    print(
        "\tF1-score (macro-avg.) w/o Ref prompts: "
        f"{np.nanmean([s for q, s in scores_by_metric['F1-score'].items() if '<vid>' not in q]):.2f}"
    )

    if output:
        results_df.to_csv(output)
        print(f"\nSaved per-query results to {output}")


def evaluate_retrieval(args):
    data_root = Path(args.data_root)

    with open(data_root / "metadata" / "queries_and_videos_test.json", "r") as f:
        gt_videos = json.load(f)

    test_video_ids, train_video_ids = load_video_universe(data_root)

    with open(args.predictions, "r") as f:
        pred_videos = json.load(f)

    errors = validate_predictions(
        pred_videos, gt_videos.keys(), test_video_ids, train_video_ids
    )
    if errors:
        raise ValueError(
            "Invalid predictions file:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    mF1, F1_queries = F1_score(gt_videos, pred_videos, sorted(test_video_ids))

    query2eco_cat, query2cv_cat = build_category_maps(
        load_query_categories(data_root), gt_videos.keys()
    )
    report({"F1-score": F1_queries}, query2eco_cat, query2cv_cat, args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Set-based retrieval --> F1-score
    retrieval = subparsers.add_parser(
        "retrieval", help="Evaluate a {query: [video_id, ...]} retrieval JSON."
    )
    retrieval.add_argument(
        "--data-root", required=True, help="Root of the downloaded HF dataset."
    )
    retrieval.add_argument(
        "--predictions",
        required=True,
        help="JSON file mapping each query to its retrieved video ids.",
    )
    retrieval.add_argument(
        "--output", help="Optional CSV path to save per-query results."
    )
    retrieval.set_defaults(func=evaluate_retrieval)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
