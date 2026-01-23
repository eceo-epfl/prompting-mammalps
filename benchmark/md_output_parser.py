"""
Parses a folder of MegaDetector outputs to form a dataframe.
"""

import json
import re
from pathlib import Path
from typing import Dict

import pandas as pd
from tqdm import tqdm


class MDVideoOutputParser:
    def __init__(self) -> None:
        self.results = {}

    def parse_file(self, json_output) -> Dict:
        """
        Parses a single JSON file and returns its content as a dictionary.
        """
        with open(json_output, "r") as f:
            content = json.load(f)

        return content

    def parse_folder(self, folder) -> Dict:
        """
        Parses all JSON files in a given folder recursively and organizes them into a nested dictionary.

        Args:
        folder (str): The folder containing JSON files to parse.

        Returns:
        Dict: A nested dictionary with parsed JSON contents.
        """
        folder_path = Path(folder)
        assert folder_path.is_dir(), f"{folder} is not a directory or can't be read."
        json_files = list(folder_path.rglob("*.json"))

        for result_file in tqdm(json_files):
            parents = result_file.relative_to(folder_path).parent.parts
            subresults = self.results
            for parent in parents:
                if parent not in subresults.keys():
                    subresults[parent] = {}
                subresults = subresults[parent]

            result_file_name = result_file.stem
            subresults[result_file_name] = self.parse_file(result_file)
            subresults[result_file_name]["detection_file_path"] = (
                result_file.relative_to(folder_path)
            )

        return self.results

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the results into a pandas DataFrame.

        Returns:
        pd.DataFrame: A DataFrame with all parsed data.
        """
        detection_df = pd.DataFrame(index=None)
        file_id_pattern = r"S\d_C\d_E\d+_V\d{4}"

        for site, site_results in self.results.items():
            for site_cam, site_cam_results in site_results.items():
                for file, file_results in site_cam_results.items():
                    results_df = pd.DataFrame.from_dict(file_results["frames"])
                    results_df["file_id"] = re.search(file_id_pattern, file).group(0)
                    results_df["detection_file_path"] = file_results[
                        "detection_file_path"
                    ]
                    # Propagate false_positive_video flag from the "info" section
                    false_positive_video_flag = file_results.get("info", {}).get(
                        "false_positive_video", None
                    )
                    if false_positive_video_flag is not None:
                        results_df["false_positive_video"] = false_positive_video_flag

                    if len(detection_df):
                        detection_df = pd.concat(
                            [detection_df, results_df], ignore_index=True
                        )
                    else:
                        detection_df = results_df

        # Separates fields of the json detection output into columns of df
        detection_exploded_df = detection_df.explode("detections")
        detections_expanded = pd.json_normalize(detection_exploded_df["detections"])

        detection_exploded_df = detection_exploded_df.reset_index(drop=True)
        detections_expanded = detections_expanded.reset_index(drop=True)

        df = pd.concat([detection_exploded_df, detections_expanded], axis=1)

        if "conf" in df.columns:
            df["conf"] = df["conf"].astype(float)

        return df
