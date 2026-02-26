"""
Zero-shot retrieval evaluation for GRAM with custom mammalps dataset.

This script:
1. Loads queries and video paths from text files
2. Extracts features using GRAM's pretrained model
3. Computes Gramian volume-based similarity matrix
4. Saves similarities for threshold-based retrieval

Adapted from CLIP4Clip eval_zeroshot.py for GRAM pipeline.
"""

import argparse
import io
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from data.audio_mapper import AudioMapper
from data.vision_mapper import VisionMapper
from easydict import EasyDict as edict
from tqdm import tqdm
from utils.build_model import build_model
from utils.logger import LOGGER
from utils.volume import volume_computation3, volume_computation4, volume_computation5

_VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi", ".mkv"}


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

    seen = set()
    video_ids = []
    video_paths = []

    for path in sorted(folder.rglob("*")):
        if (
            path.is_file()
            and path.suffix.lower() in _VIDEO_EXTENSIONS
            and path.stem not in seen
        ):
            seen.add(path.stem)
            video_ids.append(path.stem)
            video_paths.append(str(path))

    if max_videos is not None:
        video_ids = video_ids[:max_videos]
        video_paths = video_paths[:max_videos]

    return video_ids, video_paths


def extract_text_features(model, queries, device, max_len=70):
    """Extract text features using GRAM's multimodal encoder (BERT)."""

    tokenizer = model.multimodal_encoder.tokenizer

    # Tokenize all queries
    caption_tokens = tokenizer(
        queries,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    ).to(device)

    input_ids = caption_tokens.input_ids
    attention_mask = caption_tokens.attention_mask

    # Extract text features
    with torch.no_grad():
        # Get BERT embeddings
        caption_output = model.multimodal_encoder.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        # Project to contrastive space
        feat_t = model.contra_head_t(caption_output[:, 0])
        feat_t = torch.nn.functional.normalize(feat_t, p=2, dim=-1)

    return feat_t, input_ids, attention_mask


def _load_video_frames(video_path, sample_num, transforms):
    """
    Load frames from a video file by path, apply transforms.
    Returns tensor of shape (sample_num, C, H, W) or None on failure.
    Mirrors VisionMapper frame-loading logic but takes a full path directly.
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < sample_num:
            cap.release()
            return None

        frames_ids = list(range(total_frames))
        frames_split = np.array_split(frames_ids, sample_num)
        sample_idx = [s[(len(s) + 1) // 2 - 1] for s in frames_split]

        frames = []
        for idx in sample_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Pad with random frames if needed
        while len(frames) < sample_num and frames_ids:
            idx = frames_ids[len(frames_ids) // 2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames_ids = frames_ids[1:]  # avoid infinite loop

        cap.release()
        if len(frames) < sample_num:
            return None

        frames = np.array(frames[:sample_num])  # (N, H, W, 3)
        pixels = torch.from_numpy(frames.transpose(0, 3, 1, 2) / 255.0).float()
        pixels = transforms(pixels)  # (N, C, H, W)
        return pixels
    except Exception as e:
        LOGGER.warning(f"Failed to load video {video_path}: {e}")
        return None


def _load_audio_fbank(video_path, sample_num, target_length, melbins, mean, std):
    """
    Extract mel-filterbank from a video's audio track.
    Uses ffmpeg subprocess + soundfile to decode audio from any container (mp4, mkv…).
    Returns tensor of shape (sample_num, target_length, melbins) or None on failure.
    Mirrors AudioMapper BEATs loading logic but takes a full path directly.
    """
    try:
        # Decode audio via ffmpeg → 16kHz mono WAV in memory
        result = subprocess.run(
            [
                "ffmpeg", "-i", str(video_path),
                "-f", "wav", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                "-", "-nostdin", "-loglevel", "error",
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0 or len(result.stdout) == 0:
            return None

        wav_data, sr = sf.read(io.BytesIO(result.stdout))
        waveform = torch.from_numpy(wav_data).float().unsqueeze(0)  # (1, T)
        waveform = waveform - waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            num_mel_bins=melbins,
            sample_frequency=sr,
            frame_length=25,
            frame_shift=10,
        )  # (T_total, melbins)

        needed = target_length * sample_num
        src_length = fbank.shape[0]
        if src_length < needed:
            fbank = torch.nn.functional.pad(fbank, (0, 0, 0, needed - src_length))
        fbank = fbank[:needed].reshape(sample_num, target_length, melbins)
        fbank = (fbank - mean) / std
        return fbank
    except Exception as e:
        LOGGER.warning(f"Failed to load audio from {video_path}: {e}")
        return None


def extract_video_features(
    model, video_paths, vision_mapper, audio_mapper, device, batch_size=8
):
    """
    Extract video and audio features using GRAM's encoders.
    Loads frames/audio directly from full paths to avoid VisionMapper's
    non-recursive glob that fails for subdirectory-nested videos.
    Returns features for volume computation.
    """

    all_feat_v = []
    all_feat_a = []

    # Read mapper hyper-parameters (stored as .sample_num / .resolution / .melbins etc.)
    v_sample_num = vision_mapper.sample_num
    v_resolution = vision_mapper.resolution
    v_transforms = vision_mapper.transforms

    a_sample_num = audio_mapper.sample_num
    a_target_length = audio_mapper.target_length
    a_melbins = audio_mapper.melbins
    a_mean = audio_mapper.mean
    a_std = audio_mapper.std

    num_videos = len(video_paths)
    print(f"Extracting features from {num_videos} videos...")

    for i in tqdm(range(0, num_videos, batch_size)):
        batch_paths = video_paths[i : i + batch_size]

        vision_pixels_list = []
        audio_spectrograms_list = []

        for vid_path in batch_paths:
            # --- Vision ---
            vision_pixels = _load_video_frames(vid_path, v_sample_num, v_transforms)
            if vision_pixels is None:
                LOGGER.warning(f"Failed to load video: {vid_path}, using zeros")
                vision_pixels = torch.zeros(v_sample_num, 3, v_resolution, v_resolution)
            vision_pixels_list.append(vision_pixels)

            # --- Audio ---
            audio_fbank = _load_audio_fbank(
                vid_path, a_sample_num, a_target_length, a_melbins, a_mean, a_std
            )
            if audio_fbank is None:
                LOGGER.warning(f"No audio for: {vid_path}, using zeros")
                audio_fbank = torch.zeros(a_sample_num, a_target_length, a_melbins)
            audio_spectrograms_list.append(audio_fbank)

        # Stack into batches: (B, N, C, H, W) and (B, N, T, M)
        vision_pixels = torch.stack(vision_pixels_list).to(device)
        audio_spectrograms = torch.stack(audio_spectrograms_list).to(device)

        # Extract features
        with torch.no_grad():
            vision_output = model.forward_vision_encoder(vision_pixels)
            # Pool across frames/tokens before the contrastive head
            vision_output_pooled = model.pool_vision_for_contra(vision_output)
            feat_v = model.contra_head_v(vision_output_pooled)
            feat_v = torch.nn.functional.normalize(feat_v, p=2, dim=-1)

            audio_output = model.forward_audio_encoder(audio_spectrograms)
            # Pool across time/mel-tokens before the contrastive head
            audio_output_pooled = model.pool_audio_for_contra(audio_output)
            feat_a = model.contra_head_a(audio_output_pooled)
            feat_a = torch.nn.functional.normalize(feat_a, p=2, dim=-1)

            all_feat_v.append(feat_v.cpu())
            all_feat_a.append(feat_a.cpu())

    feat_v = torch.cat(all_feat_v, dim=0)
    feat_a = torch.cat(all_feat_a, dim=0)

    return feat_v, feat_a


def compute_volume_similarity(feat_t, feat_v, feat_a, feat_s=None, feat_d=None):
    """
    Compute Gramian volume-based similarity.
    Returns negative volume (lower volume = higher similarity).
    """

    if feat_s is not None:
        if feat_d is not None:
            volume = volume_computation5(feat_t, feat_v, feat_a, feat_s, feat_d)
        else:
            volume = volume_computation4(feat_t, feat_v, feat_a, feat_s)
    else:
        volume = volume_computation3(feat_t, feat_v, feat_a)

    # Return negative volume (lower volume = higher similarity)
    return -volume


def save_similarities(sim_matrix, queries, video_ids, output_dir, ground_truth_path):
    """Save similarity matrix and related metadata."""

    os.makedirs(output_dir, exist_ok=True)

    # Save raw similarity matrix
    LOGGER.info(f"Saved similarity matrix: {sim_matrix.shape}")

    # CSV: rows = videos, columns = queries
    df = pd.DataFrame(sim_matrix.T, index=video_ids, columns=queries)
    df.index.name = "video_id"
    csv_path = os.path.join(output_dir, "similarity_matrix.csv")
    df.to_csv(csv_path)
    LOGGER.info(f"Saved similarity matrix CSV to similarity_matrix.csv")

    # Save per-query results
    results = {}
    for i, query in enumerate(queries):
        query_sims = sim_matrix[i]
        results[query] = {
            "video_ids": video_ids,
            "similarities": query_sims.tolist(),
        }

    # Save metadata
    with open(os.path.join(output_dir, "queries_order.json"), "w") as f:
        json.dump(queries, f, indent=2)

    with open(os.path.join(output_dir, "video_ids.json"), "w") as f:
        json.dump(video_ids, f, indent=2)

    # Copy ground truth to output dir
    if os.path.exists(ground_truth_path):
        with open(ground_truth_path, "r") as f:
            ground_truth = json.load(f)
        with open(os.path.join(output_dir, "ground_truth.json"), "w") as f:
            json.dump(ground_truth, f, indent=2)

    # Print statistics
    LOGGER.info("\nSimilarity statistics:")
    LOGGER.info(f"  Min: {sim_matrix.min():.4f}")
    LOGGER.info(f"  Max: {sim_matrix.max():.4f}")
    LOGGER.info(f"  Mean: {sim_matrix.mean():.4f}")
    LOGGER.info(f"  Median: {np.median(sim_matrix):.4f}")
    LOGGER.info(f"  Std: {sim_matrix.std():.4f}")

    LOGGER.info(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="GRAM Zero-shot Retrieval Evaluation")

    # Dataset paths
    parser.add_argument(
        "--queries_file",
        type=str,
        required=True,
        help="Text file with one query per line",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Base directory containing videos; files are discovered recursively via rglob",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=False,
        default=None,
        help="Base directory containing audio files (if separate from video)",
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

    # Model parameters
    default_gram_root = os.environ.get("GRAM_ROOT", "models/GRAM")
    parser.add_argument(
        "--gram_root",
        type=str,
        default=default_gram_root,
        help="Path to the GRAM model root directory (used as working directory so that "
             "./pretrained_weights/... relative paths resolve correctly)",
    )
    parser.add_argument(
        "--model_cfg",
        type=str,
        default=str(
            Path(default_gram_root) / "config" / "gram" / "default_model_cfg.json"
        ),
        help="Path to model config JSON",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to GRAM checkpoint (e.g., GRAM_pretrained_4modalities)",
    )

    # Video/Audio parameters
    parser.add_argument(
        "--vision_sample_num",
        type=int,
        default=8,
        help="Number of frames to sample from video (must match checkpoint, typically 8)",
    )
    parser.add_argument(
        "--audio_sample_num",
        type=int,
        default=1,
        help="Number of audio segments to sample (must match checkpoint, typically 1)",
    )
    parser.add_argument(
        "--vision_resolution", type=int, default=224, help="Video frame resolution"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for feature extraction"
    )

    # Other parameters
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for computation",
    )

    args = parser.parse_args()

    # Resolve all user-supplied paths to absolute BEFORE changing directory,
    # because the GRAM model code loads pretrained weights via relative paths
    # (e.g. ./pretrained_weights/clip/…) which only work from inside models/GRAM.
    args.queries_file = str(Path(args.queries_file).resolve())
    args.video_path = str(Path(args.video_path).resolve())
    args.ground_truth = str(Path(args.ground_truth).resolve())
    args.output_dir = str(Path(args.output_dir).resolve())
    args.model_cfg = str(Path(args.model_cfg).resolve())
    args.checkpoint = str(Path(args.checkpoint).resolve())
    if args.audio_path:
        args.audio_path = str(Path(args.audio_path).resolve())
    gram_root = str(Path(args.gram_root).resolve())

    LOGGER.info(f"Changing working directory to GRAM root: {gram_root}")
    os.chdir(gram_root)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using device: {device}")

    # Load model config — prefer the checkpoint's own hps.json so that
    # max_vision_sample_num, max_audio_sample_num, model_type, etc. are correct.
    hps_path = Path(args.checkpoint) / "log" / "hps.json"
    if hps_path.exists():
        LOGGER.info(f"Loading model config from checkpoint hps.json: {hps_path}")
        with open(hps_path, "r") as f:
            hps = json.load(f)
        model_cfg = edict(hps["model_cfg"])
        # The checkpoint may have been trained under a different model_type name
        # (e.g. "vast"); force the local registry key "gram" so build_model works.
        model_cfg.model_type = "gram"
    else:
        LOGGER.info(f"Loading model config from: {args.model_cfg}")
        with open(args.model_cfg, "r") as f:
            model_cfg = edict(json.load(f))

    # Build model
    LOGGER.info(f"Loading model from {args.checkpoint}")
    model_cfg.pretrain_dir = args.checkpoint

    # Create a minimal args object for build_model (no DDP, local_rank=0)
    model_args = edict(
        {
            "model_cfg": model_cfg,
            "run_cfg": edict(
                {
                    "checkpoint": "",
                    "pretrain_dir": args.checkpoint,
                    "resume": False,
                    "use_ddp": False,
                }
            ),
            "local_rank": 0,
        }
    )

    model, _, _ = build_model(model_args)
    model.to(device)
    model.eval()
    LOGGER.info("Model loaded successfully")

    # Load queries
    LOGGER.info(f"Loading queries from {args.queries_file}")
    with open(args.queries_file, "r") as f:
        queries = [line.strip() for line in f if line.strip()]
    LOGGER.info(f"Loaded {len(queries)} queries")

    # Discover video files via rglob
    LOGGER.info(f"Discovering videos in {args.video_path}")
    video_ids, video_paths = discover_videos(args.video_path, args.max_videos)
    LOGGER.info(f"Discovered {len(video_ids)} videos")

    # Check if videos exist (rglob guarantees existence, but log anyway)
    missing_videos = [
        vid for vid, vpath in zip(video_ids, video_paths) if not os.path.exists(vpath)
    ]
    if missing_videos:
        LOGGER.warning(f"Warning: {len(missing_videos)} discovered paths now missing")
        LOGGER.warning(f"First few: {missing_videos[:5]}")

    # Setup vision mapper (provides normalisation transforms; actual frame loading
    # is done via _load_video_frames to handle subdirectory-nested video paths)
    vision_cfg = edict(
        {
            "name": "mammalps",
            "vision": args.video_path,
            "vision_sample_num": args.vision_sample_num,
            "vision_resolution": args.vision_resolution,
            "vision_format": "video_rawvideo",
            "training": False,
        }
    )
    vision_mapper = VisionMapper(vision_cfg, model_args)

    # Setup audio mapper
    audio_path = args.audio_path if args.audio_path else args.video_path
    audio_cfg = edict(
        {
            "audio": audio_path,
            "audio_sample_num": args.audio_sample_num,
            "audio_melbins": model_cfg.audio_melbins,
            "audio_target_length": model_cfg.audio_target_length,
            "training": False,
        }
    )
    audio_mapper = AudioMapper(audio_cfg, model_args)

    # Extract text features
    LOGGER.info("Extracting text features...")
    feat_t, input_ids, attention_mask = extract_text_features(
        model, queries, device, max_len=70
    )
    LOGGER.info(f"Text features shape: {feat_t.shape}")

    # Extract video/audio features
    feat_v, feat_a = extract_video_features(
        model,
        video_paths,
        vision_mapper,
        audio_mapper,
        device,
        batch_size=args.batch_size,
    )
    LOGGER.info(f"Video features shape: {feat_v.shape}")
    LOGGER.info(f"Audio features shape: {feat_a.shape}")

    # Move features to device for volume computation
    feat_t = feat_t.to(device)
    feat_v = feat_v.to(device)
    feat_a = feat_a.to(device)

    # Compute volume-based similarity
    LOGGER.info("Computing Gramian volume similarities...")
    sim_matrix = compute_volume_similarity(feat_t, feat_v, feat_a)
    sim_matrix = sim_matrix.cpu().numpy()

    LOGGER.info(f"Similarity matrix shape: {sim_matrix.shape}")

    # Save results
    save_similarities(
        sim_matrix, queries, video_ids, args.output_dir, args.ground_truth
    )

    LOGGER.info("\n✓ Zero-shot evaluation complete!")


if __name__ == "__main__":
    main()
