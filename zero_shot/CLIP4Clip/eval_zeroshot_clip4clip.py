"""
Optimized zero-shot retrieval with separate text-only and video-only dataloaders.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models" / "CLIP4Clip"))
from dataloaders.rawvideo_util import RawVideoExtractor
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4Clip
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from transformers import AutoTokenizer, SiglipTokenizer
from util import get_logger

torch.backends.cudnn.enabled = False

global logger


def discover_videos(video_folder, max_videos=None):
    """
    Recursively discover video files under video_folder.
    Returns (video_ids, video_paths) where video_ids are file stems
    and video_paths are the corresponding absolute path strings.
    Deduplicates by stem so .mp4 and .webm of the same clip aren't counted twice.
    """
    folder = Path(video_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Video folder not found: {folder}")

    video_ids = []
    video_paths = []

    for path in sorted(folder.rglob("*.mp4")):
        if path.is_file():
            video_ids.append(path.stem)
            video_paths.append(str(path))

    if max_videos is not None:
        video_ids = video_ids[:max_videos]
        video_paths = video_paths[:max_videos]

    return video_ids, video_paths


class TextOnlyDataLoader(data_utils.Dataset):
    """
    Dataloader that ONLY processes text queries.
    No video loading or encoding - much more efficient!
    """

    def __init__(self, queries, tokenizer, max_words=32):
        self.queries = queries
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>",
            "SEP_TOKEN": "<|endoftext|>",
            "PAD_TOKEN": "[PAD]",
        }

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        sentence = self.queries[idx]

        # Tokenize text
        words = self.tokenizer.tokenize(sentence)
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
            logger.warning(
                f"Truncating query '{sentence}' to {self.max_words} tokens (was {len(words)})"
            )
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        return (
            np.array(input_ids, dtype=np.int64),
            np.array(input_mask, dtype=np.int64),
            np.array(segment_ids, dtype=np.int64),
        )


class VideoOnlyDataLoader(data_utils.Dataset):
    def __init__(
        self,
        video_paths,
        max_frames=12,
        feature_framerate=1,
        image_resolution=224,
        frame_order=0,
        slice_framepos=2,
    ):
        """video_paths: list of absolute path strings discovered via rglob."""
        self.video_paths = video_paths
        self.max_frames = max_frames
        self.frame_order = frame_order
        self.slice_framepos = slice_framepos

        self.rawVideoExtractor = RawVideoExtractor(
            framerate=feature_framerate, size=image_resolution
        )

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]

        if not os.path.exists(video_path):
            logger.warning(f"Video not found: {video_path}")
            # Return empty video tensor
            video = np.zeros(
                (
                    self.max_frames,
                    1,
                    3,
                    self.rawVideoExtractor.size,
                    self.rawVideoExtractor.size,
                ),
                dtype=np.float32,
            )
            video_mask = np.zeros(self.max_frames, dtype=np.int64)
            return video, video_mask

        try:
            # Load video
            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data["video"]

            if len(raw_video_data.shape) > 3:
                raw_video_slice = self.rawVideoExtractor.process_raw_data(
                    raw_video_data
                )

                # Sample frames
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[: self.max_frames,]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames :,]
                    else:  # uniform sampling
                        sample_indx = np.linspace(
                            0,
                            raw_video_slice.shape[0] - 1,
                            num=self.max_frames,
                            dtype=int,
                        )
                        video_slice = raw_video_slice[sample_indx,]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(
                    video_slice, frame_order=self.frame_order
                )

                # Pad to max_frames
                slice_len = video_slice.shape[0]
                video = np.zeros(
                    (
                        self.max_frames,
                        1,
                        3,
                        self.rawVideoExtractor.size,
                        self.rawVideoExtractor.size,
                    ),
                    dtype=np.float32,
                )
                video[:slice_len,] = video_slice

                video_mask = np.zeros(self.max_frames, dtype=np.int64)
                video_mask[:slice_len] = 1

                return video, video_mask
            else:
                logger.warning(f"Invalid video shape: {video_path}")
                video = np.zeros(
                    (
                        self.max_frames,
                        1,
                        3,
                        self.rawVideoExtractor.size,
                        self.rawVideoExtractor.size,
                    ),
                    dtype=np.float32,
                )
                video_mask = np.zeros(self.max_frames, dtype=np.int64)
                return video, video_mask

        except Exception as e:
            logger.error(f"Error loading {video_path}: {e}")
            video = np.zeros(
                (
                    self.max_frames,
                    1,
                    3,
                    self.rawVideoExtractor.size,
                    self.rawVideoExtractor.size,
                ),
                dtype=np.float32,
            )
            video_mask = np.zeros(self.max_frames, dtype=np.int64)
            return video, video_mask


class SigLIPTextDataLoader(data_utils.Dataset):
    """Text dataloader for SigLIP using HuggingFace tokenizer."""

    def __init__(self, queries, tokenizer, max_words=64):
        self.queries = queries
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.queries[idx],
            padding="max_length",
            max_length=self.max_words,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        # SiglipTokenizer does not return attention_mask  derive it from padding
        if "attention_mask" in encoding:
            attention_mask = encoding["attention_mask"].squeeze(0)
        else:
            attention_mask = (input_ids != self.pad_token_id).long()
        return input_ids, attention_mask


class SigLIPWrapper(torch.nn.Module):
    """Wraps HuggingFace SigLIP with a CLIP4Clip-compatible interface.

    RawVideoExtractor normalises frames with OpenAI CLIP's mean/std.
    This wrapper undoes that normalisation and re-applies SigLIP's before
    passing frames through the vision encoder.
    """

    # OpenAI CLIP normalisation used by RawVideoExtractorCV2
    _CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    _CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
    # SigLIP normalisation
    _SIGLIP_MEAN = [0.5, 0.5, 0.5]
    _SIGLIP_STD = [0.5, 0.5, 0.5]

    def __init__(self, model_name):
        super().__init__()
        from transformers import SiglipModel

        self.siglip = SiglipModel.from_pretrained(model_name)
        # Normalisation tensors — moved to device in .to()
        self._clip_mean = torch.tensor(self._CLIP_MEAN).view(1, 3, 1, 1)
        self._clip_std = torch.tensor(self._CLIP_STD).view(1, 3, 1, 1)
        self._siglip_mean = torch.tensor(self._SIGLIP_MEAN).view(1, 3, 1, 1)
        self._siglip_std = torch.tensor(self._SIGLIP_STD).view(1, 3, 1, 1)

    def to(self, device):
        self.siglip = self.siglip.to(device)
        self._clip_mean = self._clip_mean.to(device)
        self._clip_std = self._clip_std.to(device)
        self._siglip_mean = self._siglip_mean.to(device)
        self._siglip_std = self._siglip_std.to(device)
        return self

    def eval(self):
        self.siglip.eval()
        return self

    def _renorm(self, frames):
        """Convert frames from CLIP normalisation to SigLIP normalisation."""
        raw = frames * self._clip_std + self._clip_mean  # undo CLIP norm
        return (raw - self._siglip_mean) / self._siglip_std

    def get_sequence_output(self, input_ids, attention_mask):
        """Encode text. Returns [batch, 1, embed_dim]."""
        out = self.siglip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # transformers>=5.0 returns BaseModelOutputWithPooling; older returns tensor
        text_features = out.pooler_output if hasattr(out, "pooler_output") else out
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        return text_features.unsqueeze(1)  # [batch, 1, embed_dim]

    def get_visual_output(self, video, video_mask):
        """Encode video frames. Returns [batch, max_frames, embed_dim].

        video:      [batch, 1, 1, max_frames, 3, h, w]  (CLIP-normalised)
        video_mask: [batch, max_frames]
        """
        batch_size = video.shape[0]
        max_frames = video.shape[3]

        # [batch * max_frames, 3, h, w]
        frames = video.view(
            batch_size * max_frames, 3, video.shape[-2], video.shape[-1]
        )
        frames = self._renorm(frames)

        out = self.siglip.get_image_features(pixel_values=frames)
        # transformers>=5.0 returns BaseModelOutputWithPooling; older returns tensor
        image_features = out.pooler_output if hasattr(out, "pooler_output") else out
        image_features = image_features.view(batch_size, max_frames, -1)
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        return image_features  # [batch, max_frames, embed_dim]

    def get_similarity_logits(
        self, sequence_output, visual_output, input_mask, video_mask, loose_type=True
    ):
        """Cosine similarity between text and mean-pooled video embeddings.

        Returns (logits [batch_t, batch_v], None) to match CLIP4Clip's signature.
        """
        text_embeds = sequence_output.squeeze(1)  # [batch_t, embed_dim]

        # Mean-pool over valid frames
        vmask = video_mask.float().unsqueeze(-1)  # [batch_v, max_frames, 1]
        video_embeds = (visual_output * vmask).sum(1) / vmask.sum(1).clamp(min=1e-6)
        video_embeds = torch.nn.functional.normalize(video_embeds, dim=-1)

        logits = torch.matmul(text_embeds, video_embeds.T)  # [batch_t, batch_v]
        return logits, None


def get_args():
    parser = argparse.ArgumentParser(
        description="CLIP4Clip Zero-shot Retrieval (Optimized)"
    )

    # Dataset paths
    parser.add_argument(
        "--queries_file",
        type=str,
        required=True,
        help="Text file with one query per line",
    )
    parser.add_argument(
        "--features_path",
        type=str,
        required=True,
        help="Root directory containing videos; files are discovered recursively via rglob",
    )
    parser.add_argument(
        "--ground_truth", type=str, required=True, help="Ground truth JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for similarity matrices",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Limit to the first N discovered videos (useful for smoke-testing)",
    )

    # Encoder selection
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="clip",
        choices=["clip", "siglip"],
        help="Encoder backend: 'clip' uses the CLIP4Clip pipeline (default), "
        "'siglip' uses google/siglip-so400m-patch14-384 (or --siglip_model_name)",
    )
    parser.add_argument(
        "--siglip_model_name",
        type=str,
        default="google/siglip-so400m-patch14-384",
        help="HuggingFace model ID for SigLIP (used when --encoder_type siglip)",
    )

    # Model parameters (CLIP4Clip only)
    parser.add_argument(
        "--pretrained_clip_name",
        type=str,
        default="ViT-B/32",
        help="CLIP pretrained model name (used when --encoder_type clip)",
    )
    parser.add_argument(
        "--cross_model", type=str, default="cross-base", help="Cross module"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="Where to store the pre-trained models",
    )

    # Video parameters
    parser.add_argument(
        "--max_words", type=int, default=32, help="Maximum number of words in text"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=16,
        help="Maximum number of frames to sample from video",
    )
    parser.add_argument(
        "--feature_framerate",
        type=int,
        default=1,
        help="Frame rate for video sampling (1 = sample 1 frame per second)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for processing"
    )

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--local_rank", type=int, default=0, help="For distributed")
    parser.add_argument(
        "--video_dim", type=int, default=1024, help="Video feature dimension"
    )
    parser.add_argument(
        "--loose_type",
        action="store_true",
        default=True,
        help="Use loose type for similarity (default: True for zero-shot)",
    )
    parser.add_argument(
        "--sim_header",
        type=str,
        default="meanP",
        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
        help="Similarity header type (default: meanP for zero-shot)",
    )

    args = parser.parse_args()
    return args


def set_seed_logger(args):
    global logger
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    return args


def init_device(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        n_gpu = 0

    logger.info(f"Device: {device}, n_gpu: {n_gpu}")
    return device, n_gpu


def init_model(args, device):
    """Initialize CLIP4Clip model for zero-shot evaluation"""
    cache_dir = (
        args.cache_dir
        if args.cache_dir
        else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), "distributed")
    )

    model = CLIP4Clip.from_pretrained(
        args.cross_model, cache_dir=cache_dir, task_config=args
    )
    model.to(device)
    model.eval()

    logger.info(f"Model {args.pretrained_clip_name} loaded successfully")
    return model


def _run_on_single_gpu(
    model, batch_list_t, batch_list_v, batch_t_output, batch_v_output, loose_type=False
):
    """Compute similarity between text and video features"""
    sim_matrix = []
    with torch.no_grad():
        for idx1, b1 in enumerate(batch_t_output):
            input_mask, segment_ids = batch_list_t[idx1]
            sequence_output = b1
            each_row = []
            for idx2, b2 in enumerate(batch_v_output):
                video_mask, *_tmp = batch_list_v[idx2]
                visual_output = b2
                b1b2_logits, *_tmp = model.get_similarity_logits(
                    sequence_output,
                    visual_output,
                    input_mask,
                    video_mask,
                    loose_type=loose_type,
                )
                b1b2_logits = b1b2_logits.cpu().detach().numpy()
                each_row.append(b1b2_logits)
            each_row = np.concatenate(tuple(each_row), axis=-1)
            sim_matrix.append(each_row)
    return sim_matrix


def eval_and_save_similarities(args, model, query_dataloader, video_dataloader, device):
    """Evaluate model and save similarity matrix"""
    model.eval()

    with torch.no_grad():
        # Encode all queries
        batch_sequence_output_list = []
        batch_list_t = []

        for bid, batch in enumerate(query_dataloader):
            input_ids, input_mask, segment_ids = [t.to(device) for t in batch]

            # Only encode text - no video processing!
            sequence_output = model.get_sequence_output(
                input_ids, segment_ids, input_mask
            )
            batch_sequence_output_list.append(sequence_output)
            batch_list_t.append((input_mask, segment_ids))

            if (bid + 1) % 10 == 0 or bid == len(query_dataloader) - 1:
                logger.info(f"  Encoded batch {bid + 1}/{len(query_dataloader)}")

        # Encode all videos
        logger.info("Encoding videos")
        batch_visual_output_list = []
        batch_list_v = []

        for bid, batch in tqdm.tqdm(
            enumerate(video_dataloader), total=len(video_dataloader)
        ):
            video, video_mask = [t.to(device) for t in batch]

            # Reshape video to match expected shape: [batch, pair, bs, ts, channel, h, w]
            # Current shape after batching: [batch, max_frames, 1, 3, h, w]
            # Need to add pair=1 and bs=1 dimensions, and rearrange
            batch_size = video.shape[0]
            max_frames = video.shape[1]

            # Reshape from [batch, max_frames, 1, 3, h, w] to [batch, 1, 1, max_frames, 3, h, w]
            video = video.view(
                batch_size,
                1,
                1,
                max_frames,
                video.shape[-3],
                video.shape[-2],
                video.shape[-1],
            )

            visual_output = model.get_visual_output(video, video_mask)
            batch_visual_output_list.append(visual_output)
            batch_list_v.append((video_mask,))

            if (bid + 1) % 10 == 0 or bid == len(video_dataloader) - 1:
                logger.info(f"  Encoded batch {bid + 1}/{len(video_dataloader)}")

        logger.info("Computing similarity matrix")
        sim_matrix = _run_on_single_gpu(
            model,
            batch_list_t,
            batch_list_v,
            batch_sequence_output_list,
            batch_visual_output_list,
            loose_type=args.loose_type,
        )
        sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

        logger.info(f"Similarity matrix shape: {sim_matrix.shape}")
        logger.info(f"  Queries: {sim_matrix.shape[0]}, Videos: {sim_matrix.shape[1]}")

    return sim_matrix


def eval_and_save_similarities_siglip(
    args, model, query_dataloader, video_dataloader, device
):
    """Evaluate SigLIP model and return the similarity matrix.

    Mirrors eval_and_save_similarities but handles the 2-tuple text batches
    produced by SigLIPTextDataLoader (input_ids, attention_mask).
    """
    model.eval()

    with torch.no_grad():
        # Encode all queries
        logger.info("Encoding queries")
        batch_sequence_output_list = []
        batch_list_t = []

        for bid, batch in enumerate(query_dataloader):
            input_ids, attention_mask = [t.to(device) for t in batch]
            sequence_output = model.get_sequence_output(input_ids, attention_mask)
            batch_sequence_output_list.append(sequence_output)
            batch_list_t.append((attention_mask,))

            if (bid + 1) % 10 == 0 or bid == len(query_dataloader) - 1:
                logger.info(f"  Encoded text batch {bid + 1}/{len(query_dataloader)}")

        # Encode all videos
        logger.info("Encoding videos")
        batch_visual_output_list = []
        batch_list_v = []

        for bid, batch in tqdm.tqdm(
            enumerate(video_dataloader), total=len(video_dataloader)
        ):
            video, video_mask = [t.to(device) for t in batch]
            batch_size = video.shape[0]
            max_frames = video.shape[1]

            video = video.view(
                batch_size,
                1,
                1,
                max_frames,
                video.shape[-3],
                video.shape[-2],
                video.shape[-1],
            )
            visual_output = model.get_visual_output(video, video_mask)
            batch_visual_output_list.append(visual_output)
            batch_list_v.append((video_mask,))

            if (bid + 1) % 10 == 0 or bid == len(video_dataloader) - 1:
                logger.info(f"  Encoded video batch {bid + 1}/{len(video_dataloader)}")

        # Compute similarity matrix
        logger.info("Computing similarity matrix")
        sim_matrix = []
        for idx1, seq_out in enumerate(batch_sequence_output_list):
            (attn_mask,) = batch_list_t[idx1]
            each_row = []
            for idx2, vis_out in enumerate(batch_visual_output_list):
                (vmask,) = batch_list_v[idx2]
                logits, _ = model.get_similarity_logits(
                    seq_out, vis_out, attn_mask, vmask
                )
                each_row.append(logits.cpu().detach().numpy())
            each_row = np.concatenate(each_row, axis=-1)
            sim_matrix.append(each_row)

        sim_matrix = np.concatenate(sim_matrix, axis=0)
        logger.info(f"Similarity matrix shape: {sim_matrix.shape}")
        logger.info(f"  Queries: {sim_matrix.shape[0]}, Videos: {sim_matrix.shape[1]}")

    return sim_matrix


def save_similarities(sim_matrix, queries, video_ids, output_dir, ground_truth):
    """Save similarity matrix in multiple formats"""

    # CSV: rows = videos, columns = queries
    df = pd.DataFrame(sim_matrix.T, index=video_ids, columns=queries)
    df.index.name = "video_id"
    csv_path = os.path.join(output_dir, "similarity_matrix.csv")
    df.to_csv(csv_path)
    logger.info(f"Saved similarity matrix CSV to similarity_matrix.csv")

    results = {}
    for i, query in enumerate(queries):
        query_sims = sim_matrix[i]
        sorted_indices = np.argsort(-query_sims)

        results[query] = {
            "video_ids": video_ids,
            "similarities": query_sims.tolist(),
        }

    # Save metadata
    with open(os.path.join(output_dir, "queries_order.json"), "w") as f:
        json.dump(queries, f, indent=2)

    with open(os.path.join(output_dir, "video_ids.json"), "w") as f:
        json.dump(video_ids, f, indent=2)

    logger.info("\nSimilarity statistics:")
    logger.info(f"  Min: {sim_matrix.min():.4f}")
    logger.info(f"  Max: {sim_matrix.max():.4f}")
    logger.info(f"  Mean: {sim_matrix.mean():.4f}")
    logger.info(f"  Median: {np.median(sim_matrix):.4f}")
    logger.info(f"  Std: {sim_matrix.std():.4f}")


def main():
    global logger

    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args)

    # Load queries from text file
    logger.info(f"Loading queries from {args.queries_file}")
    with open(args.queries_file, "r") as f:
        queries = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(queries)} queries")

    # Discover video files via rglob
    logger.info(f"Discovering videos in {args.features_path}")
    video_ids, video_paths = discover_videos(args.features_path, args.max_videos)
    logger.info(f"Discovered {len(video_ids)} videos")

    # ------------------------------------------------------------------ #
    #  Encoder-specific setup                                              #
    # ------------------------------------------------------------------ #
    if args.encoder_type == "siglip":

        logger.info(f"Loading SigLIP model: {args.siglip_model_name}")
        tokenizer = SiglipTokenizer.from_pretrained(args.siglip_model_name)
        model = SigLIPWrapper(args.siglip_model_name)
        model.to(device)
        model.eval()
        logger.info("SigLIP model loaded successfully")

        # SigLIP expects 384×384 input and max 64 tokens
        image_resolution = 384
        max_words = min(args.max_words, 64)

        query_dataset = SigLIPTextDataLoader(
            queries=queries, tokenizer=tokenizer, max_words=max_words
        )
        eval_fn = eval_and_save_similarities_siglip

    else:  # clip (default)
        tokenizer = ClipTokenizer()
        model = init_model(args, device)
        image_resolution = 224
        query_dataset = TextOnlyDataLoader(
            queries=queries, tokenizer=tokenizer, max_words=args.max_words
        )
        eval_fn = lambda a, m, qd, vd, dev: eval_and_save_similarities(
            a, m, qd, vd, dev
        )

    query_dataloader = data_utils.DataLoader(
        query_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # Create VIDEO-ONLY dataloader (uses discovered paths directly)
    video_dataset = VideoOnlyDataLoader(
        video_paths=video_paths,
        max_frames=args.max_frames,
        feature_framerate=args.feature_framerate,
        image_resolution=image_resolution,
    )

    video_dataloader = data_utils.DataLoader(
        video_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # Load ground truth
    with open(args.ground_truth, "r") as f:
        ground_truth = json.load(f)

    sim_matrix = eval_fn(args, model, query_dataloader, video_dataloader, device)
    save_similarities(sim_matrix, queries, video_ids, args.output_dir, ground_truth)


if __name__ == "__main__":
    main()
