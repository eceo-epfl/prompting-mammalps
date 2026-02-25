from typing import Optional, Union

import numpy as np
from sklearn.metrics import average_precision_score


def mean_average_precision_score(gt_queries_videos, sim_matrix_df):
    """Computes mAP for a set of query and predicted relevance score to each query"""
    APi = {}
    video_ids = list(sim_matrix_df.index)
    for query in gt_queries_videos.keys():
        true = np.array(
            [v in gt_queries_videos[query]["videos"] for v in video_ids]
        ).astype(int)
        if np.sum(true) == 0:
            # Query never not associated to any video.
            APi[query] = np.nan
            continue
        if query in sim_matrix_df.columns:
            scores = sim_matrix_df[
                query
            ].values  # Cosine similarity between this query and all candidate videos
            AveP = average_precision_score(true, scores)
            APi[query] = AveP
        else:
            print(f"Query {query} has not been processed")
            APi[query] = np.nan

    if len(gt_queries_videos):
        return np.nanmean(list(APi.values())), APi
    else:
        return np.nan, APi


def F1_score_from_sim(
    gt_queries_videos,
    sim_matrix_df,
    best_thresholds: Optional[Union[dict, float]] = None,
):
    F1i = {}
    if isinstance(best_thresholds, float):
        base_threshold = best_thresholds
    else:
        base_threshold = np.median(sim_matrix_df.values)

    for q in gt_queries_videos.keys():
        if q not in sim_matrix_df.columns:
            print(f"Query {q} has not been processed")
            continue
        if isinstance(best_thresholds, dict) and q in best_thresholds:
            t_q = best_thresholds[q]
        else:
            t_q = base_threshold
        gt_videos_q = list(gt_queries_videos[q]["videos"])
        ass_videos_q = list(sim_matrix_df.index[sim_matrix_df[q] >= t_q])

        F1i[q] = F1_score_q(gt_videos_q, ass_videos_q)

    return np.nanmean(list(F1i.values())), F1i


def IoU_from_sim(
    gt_queries_videos,
    sim_matrix_df,
    best_thresholds: Optional[Union[dict, float]] = None,
):
    IoUi = {}
    if isinstance(best_thresholds, float):
        base_threshold = best_thresholds
    else:
        base_threshold = np.median(sim_matrix_df.values)

    for q in gt_queries_videos.keys():
        if q not in sim_matrix_df.columns:
            print(f"Query {q} has not been processed")
            continue
        if isinstance(best_thresholds, dict) and q in best_thresholds:
            t_q = best_thresholds[q]
        else:
            t_q = base_threshold
        gt_videos_q = list(gt_queries_videos[q]["videos"])
        ass_videos_q = list(sim_matrix_df.index[sim_matrix_df[q] >= t_q])

        IoUi[q] = IoU_q(gt_videos_q, ass_videos_q)

    return np.nanmean(list(IoUi.values())), IoUi


def F1_score_q(gt_videos, pred_videos) -> float:
    tp = len([v for v in pred_videos if v in gt_videos])
    fp = len([v for v in pred_videos if v not in gt_videos])
    fn = len([v for v in gt_videos if (v not in pred_videos)])

    if 2 * tp + fp + fn == 0:
        f1_score = np.nan
    else:
        f1_score = 2 * tp / (2 * tp + fp + fn)

    return f1_score


def IoU_q(gt_videos, pred_videos) -> float:
    union = len(set(gt_videos).union(set(pred_videos)))
    intersection = len(set(gt_videos).intersection(set(pred_videos)))

    if union != 0:
        return intersection / union
    else:
        return np.nan


def F1_score(gt_queries_videos, pred_queries_videos) -> tuple:

    F1i = {}

    for q in gt_queries_videos.keys():
        gt_videos_q = list(gt_queries_videos[q])
        ass_videos_q = list(pred_queries_videos[q])

        F1i[q] = F1_score_q(gt_videos_q, ass_videos_q)

    return np.nanmean(list(F1i.values())), F1i


def IoU(gt_queries_videos, pred_queries_videos) -> tuple:

    IoUi = {}

    for q in gt_queries_videos.keys():
        gt_videos_q = list(gt_queries_videos[q])
        ass_videos_q = list(pred_queries_videos[q])

        IoUi[q] = IoU_q(gt_videos_q, ass_videos_q)

    return np.nanmean(list(IoUi.values())), IoUi
