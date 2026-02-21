"""
Convert queries_and_videos_check.json to CLIP4Clip multi-sentence format.

This handles the many-to-many mapping problem where:
- One query matches multiple videos
- One video matches multiple queries (e.g., S3_C2_E515_V0086)

Uses the same approach as MSVD dataset with multi_sentence_per_video=True.
"""

import argparse
import json
import os


def convert_to_multisentence_format(input_json_path, output_dir, output_name="test"):
    """
    Convert queries_and_videos_check.json to CLIP4Clip multi-sentence format.

    Creates:
    - {output_name}.csv: video_id,sentence pairs (with duplicates for many-to-many)
    - metadata.json: contains cut_off_points, query mappings, video list

    Args:
        input_json_path: Path to input JSON file
        output_dir: Output directory
        output_name: Name for output CSV ('train' or 'test')
    """
    output_dir = os.path.join(output_dir, output_name)
    with open(input_json_path, "r") as f:
        data = json.load(f)

    # Flatten to sentence-video pairs
    sentences_dict = {}
    cut_off_points = []
    query_to_videos = {}

    sentence_idx = 0
    query_idx = 0

    print("Processing categories")

    # Process each category
    for category, queries in data.items():
        if not queries:  # Skip empty categories
            print(f"  Skipping empty category: {category}")
            continue

        print(f"\n  Category: {category}")

        for query_text, video_ids in queries.items():
            if not video_ids:  # Skip empty queries
                print(f"    Skipping empty query: {query_text}")
                continue

            print(f"    Query {query_idx}: '{query_text}' → {len(video_ids)} videos")

            query_to_videos[query_idx] = {
                "query": query_text,
                "category": category,
                "video_ids": video_ids,
                "start_idx": sentence_idx,
                "end_idx": sentence_idx + len(video_ids),
            }

            # Add all video-sentence pairs for this query
            for video_id in video_ids:
                sentences_dict[sentence_idx] = {
                    "video_id": video_id,
                    "sentence": query_text,
                    "query_idx": query_idx,
                    "category": category,
                }
                sentence_idx += 1

            # Mark the cutoff point (cumulative count)
            cut_off_points.append(sentence_idx)
            query_idx += 1

    # Get unique videos (order matters for similarity matrix)
    # Preserve order of first appearance
    all_videos_ordered = []
    seen_videos = set()
    for idx in range(len(sentences_dict)):
        vid = sentences_dict[idx]["video_id"]
        if vid not in seen_videos:
            all_videos_ordered.append(vid)
            seen_videos.add(vid)

    # Create video_id to index mapping
    video_id_to_idx = {vid: idx for idx, vid in enumerate(all_videos_ordered)}

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save CSV (format: video_id,sentence)
    csv_path = os.path.join(output_dir, f"{output_name}.csv")
    with open(csv_path, "w") as f:
        f.write("video_id,sentence\n")
        for idx in range(len(sentences_dict)):
            vid = sentences_dict[idx]["video_id"]
            sent = sentences_dict[idx]["sentence"]
            # Escape commas in sentences
            sent_escaped = sent.replace(",", ";")
            f.write(f"{vid},{sent_escaped}\n")

    # Save metadata for evaluation
    metadata = {
        "sentence_num": len(sentences_dict),
        "video_num": len(all_videos_ordered),
        "query_num": len(query_to_videos),
        "cut_off_points": cut_off_points,
        "query_to_videos": query_to_videos,
        "video_id_to_idx": video_id_to_idx,
        "all_videos": all_videos_ordered,
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save ground truth in simple format (for custom evaluation)
    ground_truth = {}
    for qid, info in query_to_videos.items():
        ground_truth[info["query"]] = {
            "videos": info["video_ids"],
            "category": info["category"],
        }

    gt_path = os.path.join(output_dir, "ground_truth.json")
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    # Save unique queries to queries.txt (for eval_zeroshot.py)
    queries_txt_path = os.path.join(output_dir, "queries.txt")
    with open(queries_txt_path, "w") as f:
        for i in range(len(query_to_videos)):
            f.write(query_to_videos[i]["query"] + "\n")

    # Save unique videos to video_ids.txt (for eval_zeroshot.py)
    video_ids_txt_path = os.path.join(output_dir, "video_ids.txt")
    with open(video_ids_txt_path, "w") as f:
        for vid in all_videos_ordered:
            f.write(vid + "\n")

    print(f"Created {output_name}.csv with {len(sentences_dict)} sentence-video pairs")
    video_query_count = {}
    for idx in range(len(sentences_dict)):
        vid = sentences_dict[idx]["video_id"]
        qid = sentences_dict[idx]["query_idx"]
        if vid not in video_query_count:
            video_query_count[vid] = []
        if qid not in video_query_count[vid]:
            video_query_count[vid].append(qid)

    multi_query_videos = {k: v for k, v in video_query_count.items() if len(v) > 1}

    if multi_query_videos:
        print(f"\nFound {len(multi_query_videos)} videos matching multiple queries:")
        for vid, qids in sorted(multi_query_videos.items()):
            queries = [query_to_videos[qid]["query"] for qid in qids]
            print(f"  {vid}:")
            for q in queries:
                print(f"    - '{q}'")
    else:
        print("No videos match multiple queries (one-to-one mapping).")

    # Statistics per category
    category_stats = {}
    for qid, info in query_to_videos.items():
        cat = info["category"]
        if cat not in category_stats:
            category_stats[cat] = {"queries": 0, "videos": 0}
        category_stats[cat]["queries"] += 1
        category_stats[cat]["videos"] += len(info["video_ids"])

    for cat, stats in sorted(category_stats.items()):
        print(
            f"  {cat}: {stats['queries']} queries, {stats['videos']} video-query pairs"
        )

    return metadata


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert wildlife dataset to multi-sentence format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="queries_and_videos_check.json",
        help="Input JSON file with query-video mappings",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="clip4clip/dataset",
        help="Output directory for dataset files",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Output CSV name: train.csv or test.csv",
    )

    args = parser.parse_args()

    # Convert format
    metadata = convert_to_multisentence_format(
        args.input, args.output_dir, args.output_name
    )
