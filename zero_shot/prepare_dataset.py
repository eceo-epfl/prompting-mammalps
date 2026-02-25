import argparse
import csv
import json
import os
from pathlib import Path


def _discover_videos(video_folder, max_videos=None):
    """
    Recursively discover video files under video_folder.
    Returns a sorted list of video ID stems (e.g. 'S1_C1_E1_V0001').
    Deduplicates by stem so .mp4 and .webm of the same clip aren't counted twice.
    Pass max_videos to limit the pool (useful for smoke-testing).
    """
    folder = Path(video_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Video folder not found: {folder}")

    seen = set()
    video_ids = []
    for path in sorted(folder.rglob("*")):
        if (
            path.is_file()
            and path.suffix.lower() in {".mp4", ".webm", ".avi", ".mkv"}
            and path.stem not in seen
        ):
            seen.add(path.stem)
            video_ids.append(path.stem)

    if max_videos is not None:
        video_ids = video_ids[:max_videos]

    print(f"Discovered {len(video_ids)} videos in {folder}")
    return video_ids


def prepare_dataset(input_json_path, output_dir, video_folder, max_videos=None):
    """
    Parse input_json_path and write queries.txt + ground_truth.json to output_dir.
    Also discovers videos in video_folder (for stats / smoke-test limiting).
    """
    with open(input_json_path) as f:
        data = json.load(f)

    queries = []
    ground_truth = {}

    for category, category_queries in data.items():
        if not category_queries or category == "VIDEO_COMPARISON":
            print(f"Skipping category: {category}")
            continue
        for query_text, video_ids in category_queries.items():
            if not video_ids:
                print(f"Skipping empty query: {query_text}")
                continue
            queries.append(query_text)
            ground_truth[query_text] = {"videos": video_ids, "category": category}

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)

    with open(os.path.join(output_dir, "queries.txt"), "w") as f:
        for query in queries:
            f.write(query + "\n")

    # Write pairs.csv: one row per (video_id, sentence) — used by training_clip4clip.py
    pairs_path = os.path.join(output_dir, "pairs.csv")
    with open(pairs_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "sentence"])
        writer.writeheader()
        for query_text, info in ground_truth.items():
            for vid in info["videos"]:
                writer.writerow({"video_id": vid, "sentence": query_text})

    discovered = _discover_videos(video_folder, max_videos)

    num_pairs = sum(len(info["videos"]) for info in ground_truth.values())
    print(f"Queries: {len(queries)}, Pairs: {num_pairs}")
    print(f"Videos discovered (from folder): {len(discovered)}")
    return {
        "query_num": len(queries),
        "video_num": len(discovered),
        "pair_num": num_pairs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Write queries.txt + ground_truth.json to one or more output dirs"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input JSON with query-video mappings"
    )
    parser.add_argument(
        "--output_dirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more output directories (e.g. zero_shot/CLIP4Clip/dataset/test zero_shot/GRAM/dataset/test)",
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        required=True,
        help="Root folder containing videos; discovered recursively via rglob",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Limit to the first N discovered videos (useful for smoke-testing)",
    )

    args = parser.parse_args()

    for output_dir in args.output_dirs:
        print(f"\n--- Preparing {output_dir} ---")
        prepare_dataset(
            input_json_path=args.input,
            output_dir=output_dir,
            video_folder=args.video_folder,
            max_videos=args.max_videos,
        )


if __name__ == "__main__":
    main()
