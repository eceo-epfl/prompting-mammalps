"""
Training Workflow:
1. Load pre-trained CLIP4Clip model (ViT-B/32)
2. Create paired video-text data from  queries_and_videos_check.json
3. Forward pass: Encode videos and texts separately
4. Compute contrastive loss: Videos should be close to their paired texts
5. Backward pass: Update model weights
6. Repeat for N epochs

Loss Function:
- Bidirectional contrastive loss (InfoNCE)
- Video-to-text: Each video should match its text among all texts in batch
- Text-to-video: Each text should match its video among all videos in batch
- This encourages the model to learn discriminative video-text alignments
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import math
import os
import random
from collections import Counter
from typing import Iterator

import numpy as np
import torch
import torch.utils.data as data_utils
import tqdm
from dataloaders.rawvideo_util import RawVideoExtractor
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4Clip
from modules.optimization import BertAdam
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from PIL import Image, ImageDraw, ImageFont
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from util import get_logger

global logger
global tb_writer


class BalancedDistributedSampler(DistributedSampler):
    """
    Distributed sampler with balanced/weighted sampling to handle caption imbalance.

    Strategy:
    1. Count frequency of each caption in the dataset
    2. Compute weight for each sample as 1 / caption_frequency
    3. Use torch.multinomial to sample indices proportional to weights
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
        replacement=True,
    ):
        """
        Args:
            dataset: Dataset with .pairs attribute containing {'query': str, 'video_id': str}
            num_replicas: Number of processes (GPUs)
            rank: Rank of current process
            shuffle: Whether to use random sampling (always True for balanced sampling)
            seed: Random seed
            drop_last: Whether to drop incomplete batch
            replacement: Whether to allow replacement in multinomial sampling
        """
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

        self.replacement = replacement

        # Compute weights for balanced sampling
        self.sample_weights = self._compute_sample_weights(dataset)

        logger.info(f"BalancedDistributedSampler initialized:")
        logger.info(f"  Total samples: {len(dataset)}")
        logger.info(f"  Sample weights computed from caption frequencies")
        logger.info(f"  Min weight: {self.sample_weights.min().item():.6f}")
        logger.info(f"  Max weight: {self.sample_weights.max().item():.6f}")
        logger.info(f"  Mean weight: {self.sample_weights.mean().item():.6f}")

    def _compute_sample_weights(self, dataset):
        """
        Compute inverse frequency weights for each sample.

        Process:
        1. Count frequency of each unique caption
        2. Assign weight = 1 / frequency to each sample
        3. Normalize weights to sum to 1.0

        Returns:
            torch.Tensor: Weights for each sample in [0, 1]
        """
        # Extract captions from dataset pairs
        captions = [pair["query"] for pair in dataset.pairs]

        # Count caption frequencies
        caption_counts = Counter(captions)

        # Compute inverse frequency weights
        # weight[i] = 1 / count[caption[i]]
        sample_weights = torch.zeros(len(captions), dtype=torch.float32)
        for i, caption in enumerate(captions):
            sample_weights[i] = 1.0 / caption_counts[caption]

        # Normalize weights to sum to 1 (required for multinomial)
        sample_weights = sample_weights / sample_weights.sum()

        return sample_weights

    def __iter__(self) -> Iterator[int]:
        """
        Build a random iterator with balanced/weighted sampling.

        Uses torch.multinomial to sample indices proportional to inverse
        caption frequencies. This ensures each caption is equally represented.

        Returns:
            Iterator[int]: Iterator over sample indices.
        """
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Sample indices according to caption frequencies (with or without replacement)
        indices = torch.multinomial(
            self.sample_weights, len(self.dataset), self.replacement, generator=g
        ).tolist()

        if not self.drop_last:
            # Add extra samples to make it evenly divisible by num_replicas
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # Remove tail of data to make it evenly divisible
            indices = indices[: self.total_size]

        assert (
            len(indices) == self.total_size
        ), f"Indices length {len(indices)} != total_size {self.total_size}"

        # Subsample to give each GPU its portion
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert (
            len(indices) == self.num_samples
        ), f"Subsampled indices length {len(indices)} != num_samples {self.num_samples}"

        return iter(indices)


def custom_collate_fn(batch):
    """
    Custom collate function that handles mixed types (tensors and strings).
    Keeps strings as lists instead of trying to stack them.
    """
    # Separate tensors from metadata
    input_ids_list = [item[0] for item in batch]
    input_mask_list = [item[1] for item in batch]
    segment_ids_list = [item[2] for item in batch]
    video_list = [item[3] for item in batch]
    video_mask_list = [item[4] for item in batch]
    video_ids = [item[5] for item in batch]  # Keep as list
    queries = [item[6] for item in batch]  # Keep as list

    # Stack tensors
    return (
        torch.stack(input_ids_list),
        torch.stack(input_mask_list),
        torch.stack(segment_ids_list),
        torch.stack(video_list),
        torch.stack(video_mask_list),
        video_ids,
        queries,
    )


def save_frame_with_text(frame_tensor, video_id, caption, output_dir, frame_idx=0):
    """
    Save a video frame as JPG with text overlay showing frame_id and caption.

    Args:
        frame_tensor: Tensor of shape [3, H, W] with normalized values
        video_id: Video ID string
        caption: Caption string
        output_dir: Directory to save the frame
        frame_idx: Frame index in the video
    """
    # Inverse normalization (CLIP normalization values)
    # Original normalization: (img - mean) / std
    # Inverse: img * std + mean
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    frame_denorm = frame_tensor * std + mean
    frame_denorm = torch.clamp(frame_denorm, 0, 1)  # Clamp to [0, 1]

    # Convert tensor to numpy and denormalize
    frame_np = frame_denorm.cpu().numpy()  # [3, H, W]
    frame_np = np.transpose(frame_np, (1, 2, 0))  # [H, W, 3]
    frame_np = (frame_np * 255).astype(np.uint8)

    # Create PIL image
    img = Image.fromarray(frame_np)

    # Add text overlay
    draw = ImageDraw.Draw(img)
    text = f"{caption}_frame_{frame_idx}"

    # Try to use a reasonable font size, fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 8)
    except:
        font = ImageFont.load_default()

    # Add text with background for readability
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Draw background rectangle
    margin = 5
    draw.rectangle(
        [(0, 0), (text_width + 2 * margin, text_height + 2 * margin)], fill=(0, 0, 0)
    )
    # Draw text
    draw.text((margin, margin), text, fill=(255, 255, 255), font=font)

    # Save as JPG
    filename = f"{video_id}_frame_{frame_idx}.jpg"
    filepath = os.path.join(output_dir, filename)
    img.save(filepath, "JPEG", quality=95)


class MammAlpsDataLoader(data_utils.Dataset):
    """
    MammAlps dataset loader for training.

    Loads query-video pairs from CSV file (output of prepare_clip4clip_dataset.py).
    Each pair consists of:
    - Text query (e.g., "An animal bathing")
    - Video ID (e.g., "S3_C3_E524_V0327")
    - Video frames extracted from videos/test/S#/C#/video_id.mp4

    CSV format:
    video_id,sentence
    S3_C3_E524_V0327,An animal bathing
    S3_C2_E710_V0245,An animal bathing
    ...
    """

    def __init__(
        self,
        csv_path,
        video_dir,
        tokenizer,
        max_words=32,
        max_frames=12,
        feature_framerate=1,
        image_resolution=224,
        frame_order=0,
        slice_framepos=2,
    ):
        """
        Args:
            csv_path: Path to CSV file (train.csv or test.csv)
            video_dir: Root directory containing videos (e.g., videos/train/)
            tokenizer: CLIP tokenizer
            max_words: Maximum text tokens
            max_frames: Maximum video frames to sample
            feature_framerate: Frames per second to sample (1 = 1 fps)
            image_resolution: Input image size
            frame_order: 0=sequential, 1=reverse, 2=random
            slice_framepos: 0=head, 1=tail, 2=uniform sampling
        """
        self.video_dir = video_dir
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.max_frames = max_frames
        self.frame_order = frame_order
        self.slice_framepos = slice_framepos

        # Initialize video extractor
        self.rawVideoExtractor = RawVideoExtractor(
            framerate=feature_framerate, size=image_resolution
        )

        # Special tokens for CLIP
        self.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>",
            "SEP_TOKEN": "<|endoftext|>",
            "PAD_TOKEN": "[PAD]",
        }

        # Load and prepare data
        self.pairs = self._load_csv(csv_path)

        logger.info(f"MammAlpsDataLoader initialized:")
        logger.info(f"  Total pairs: {len(self.pairs)}")
        logger.info(f"  Video directory: {video_dir}")
        logger.info(f"  Max words: {max_words}, Max frames: {max_frames}")
        logger.info(f"  Feature framerate: {feature_framerate} fps")

    def _load_csv(self, csv_path):
        """
        Load CSV file and create list of (query, video_id) pairs.

        CSV format:
        video_id,sentence
        S3_C3_E524_V0327,An animal bathing

        Returns:
            List of dicts: [{'query': str, 'video_id': str}, ...]
        """
        import csv

        pairs = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row["video_id"].strip()
                query = row["sentence"].strip()
                # Restore commas that were escaped as semicolons
                query = query.replace(";", ",")

                pairs.append(
                    {
                        "query": query,
                        "video_id": video_id,
                    }
                )

        logger.info(f"Loaded {len(pairs)} query-video pairs from {csv_path}")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Get a single training sample.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (input_ids, input_mask, segment_ids, video, video_mask)
        """
        pair = self.pairs[idx]
        query = pair["query"]
        video_id = pair["video_id"]

        # Get text features
        input_ids, input_mask, segment_ids = self._get_text(query)

        # Get video features
        try:
            video, video_mask = self._get_video(video_id)
        except Exception as e:
            # If video loading fails, return zero tensors
            logger.warning(f"Failed to load video {video_id}: {e}")
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

        # Convert to tensors and include metadata
        return (
            torch.from_numpy(input_ids),
            torch.from_numpy(input_mask),
            torch.from_numpy(segment_ids),
            torch.from_numpy(video),
            torch.from_numpy(video_mask),
            video_id,  # Keep as string
            query,  # Keep as string
        )

    def _get_text(self, query):
        """
        Tokenize text query.

        Process:
        1. Tokenize: "An animal bathing" -> ["an", "animal", "bathing"]
        2. Add special tokens: ["<|startoftext|>", "an", "animal", "bathing", "<|endoftext|>"]
        3. Convert to IDs: [49406, 550, 3957, 17983, 49407]
        4. Pad to max_words

        Returns:
            input_ids: [max_words] token IDs
            input_mask: [max_words] attention mask (1=real token, 0=padding)
            segment_ids: [max_words] segment IDs (all 0 for single sentence)
        """
        words = self.tokenizer.tokenize(query)

        # Add special tokens
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

        # Convert to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        # Pad to max_words
        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        return (
            np.array(input_ids, dtype=np.int64),
            np.array(input_mask, dtype=np.int64),
            np.array(segment_ids, dtype=np.int64),
        )

    def _get_video_path(self, video_id):
        """
        Convert video_id to nested S#/C#/ path structure. S3_C2_E515_V0086" -> "videos/test/S3/C2/S3_C2_E515_V0086.mp4"
        """
        parts = video_id.split("_")
        if len(parts) >= 2:
            s_part = parts[0]
            c_part = parts[1]
            video_path = os.path.join(self.video_dir, s_part, c_part, f"{video_id}.mp4")
        else:
            video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        return video_path

    def _get_video(self, video_id):
        """
        Load and process video frames.

        Process:
        1. Load video file (.mp4 or .webm)
        2. Extract raw frames at specified framerate
        3. Sample max_frames from extracted frames
        4. Normalize pixels to [0, 1] and resize to image_resolution
        5. Pad to max_frames if needed

        Returns:
            video: [max_frames, 1, 3, H, W] video tensor (float32)
            video_mask: [max_frames] mask (1=valid frame, 0=padding)
        """
        video_path = self._get_video_path(video_id)

        # Load video
        raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
        raw_video_data = raw_video_data["video"]

        if len(raw_video_data.shape) > 3:
            # Process raw frames
            raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data)

            # Sample frames to max_frames
            if self.max_frames < raw_video_slice.shape[0]:
                if self.slice_framepos == 0:
                    # Take first max_frames
                    video_slice = raw_video_slice[: self.max_frames, ...]
                elif self.slice_framepos == 1:
                    # Take last max_frames
                    video_slice = raw_video_slice[-self.max_frames :, ...]
                else:
                    # Uniform sampling
                    sample_indx = np.linspace(
                        0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int
                    )
                    video_slice = raw_video_slice[sample_indx, ...]
            else:
                video_slice = raw_video_slice

            # Apply frame order
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
            video[:slice_len, ...] = video_slice

            video_mask = np.zeros(self.max_frames, dtype=np.int64)
            video_mask[:slice_len] = 1

            return video, video_mask


def get_args():
    parser = argparse.ArgumentParser(description="CLIP4Clip Fine-tuning on MammAlps")

    # Dataset paths
    parser.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="Path to training CSV file (e.g., clip4clip/train/train.csv)",
    )
    parser.add_argument(
        "--train_video_dir",
        type=str,
        required=True,
        help="Root directory containing training videos",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=None,
        help="Path to test/validation CSV file (optional, e.g., clip4clip/test/test.csv)",
    )
    parser.add_argument(
        "--test_video_dir",
        type=str,
        default=None,
        help="Root directory containing test videos (optional)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="mammalps_checkpoints",
        help="Output directory for model checkpoints",
    )

    # Model parameters
    parser.add_argument(
        "--pretrained_clip_name",
        type=str,
        default="ViT-B/32",
        help="CLIP pretrained model name",
    )
    parser.add_argument(
        "--cross_model",
        type=str,
        default="cross-base",
        help="Cross module architecture",
    )
    parser.add_argument(
        "--cache_dir", type=str, default="", help="Where to store pre-trained models"
    )
    parser.add_argument(
        "--init_model",
        type=str,
        default=None,
        help="Path to initial model checkpoint (optional)",
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Training batch size (reduce if OOM)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate (1e-5 for fine-tuning)"
    )
    parser.add_argument(
        "--warmup_proportion",
        type=float,
        default=0.1,
        help="Proportion of training for learning rate warmup",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Accumulate gradients over N steps",
    )
    parser.add_argument(
        "--coef_lr",
        type=float,
        default=1e-3,
        help="Coefficient for CLIP learning rate (1e-3 = slower)",
    )
    parser.add_argument("--n_display", type=int, default=10, help="Log every N steps")

    # Video/Text parameters
    parser.add_argument(
        "--max_words", type=int, default=32, help="Maximum number of words in text"
    )
    parser.add_argument(
        "--max_frames", type=int, default=12, help="Maximum number of frames to sample"
    )
    parser.add_argument(
        "--feature_framerate",
        type=int,
        default=1,
        help="Frame sampling rate (1 = 1 fps)",
    )
    parser.add_argument(
        "--image_resolution", type=int, default=224, help="Image resolution"
    )

    # Architecture parameters
    parser.add_argument(
        "--loose_type",
        action="store_true",
        default=False,
        help="Use loose type (False for training with cross-attention)",
    )
    parser.add_argument(
        "--sim_header",
        type=str,
        default="tightTransf",
        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
        help="Similarity header (tightTransf for training with cross-attention)",
    )
    parser.add_argument(
        "--text_num_hidden_layers",
        type=int,
        default=12,
        help="Number of text encoder layers",
    )
    parser.add_argument(
        "--visual_num_hidden_layers",
        type=int,
        default=12,
        help="Number of visual encoder layers",
    )
    parser.add_argument(
        "--cross_num_hidden_layers",
        type=int,
        default=4,
        help="Number of cross-attention layers",
    )

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--video_dim", type=int, default=1024, help="Video feature dimension"
    )
    parser.add_argument(
        "--freeze_layer_num",
        type=int,
        default=-1,
        help="Freeze first N layers (-1 = train all)",
    )
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs to use")

    # Debug option
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode: print batch info (video IDs, captions) at every step",
    )

    # Distributed training parameter
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="For distributed training: local rank (automatically set by launcher)",
    )

    args = parser.parse_args()

    # Set task_type for model initialization
    args.task_type = "retrieval"
    args.datatype = "mammalps"

    return args


def set_seed_logger(args):
    """Set random seeds and initialize logger."""
    global logger

    # Set seeds for reproducibility
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Initialize distributed training if using multiple GPUs
    # Check if running in distributed mode
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.rank = int(os.environ.get("RANK", 0))
        args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))

        # Initialize process group if in distributed mode
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        # Get values from PyTorch
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        torch.cuda.set_device(args.local_rank)
    else:
        # Single GPU mode
        args.world_size = 1
        args.rank = 0

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize logger
    logger = get_logger(os.path.join(args.output_dir, "training_log.txt"))

    if args.local_rank == 0:
        logger.info("CLIP4Clip Fine-tuning on MammAlps Dataset")
        logger.info("\nArguments:")
        for key in sorted(args.__dict__):
            logger.info(f"  {key}: {args.__dict__[key]}")

    return args


def init_device(args, local_rank):
    """Initialize device (GPU or CPU)."""
    global logger

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        n_gpu = 0

    if logger and args.local_rank == 0:
        logger.info(f"device: {device} n_gpu: {n_gpu}")

    args.n_gpu = n_gpu
    return device, n_gpu


def init_model(args, device, n_gpu, local_rank):
    """Initialize CLIP4Clip model."""
    global logger

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location="cpu")
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = (
        args.cache_dir
        if args.cache_dir
        else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), "distributed")
    )
    model = CLIP4Clip.from_pretrained(
        args.cross_model,
        cache_dir=cache_dir,
        state_dict=model_state_dict,
        task_config=args,
    )

    model.to(device)

    for param in model.clip.transformer.parameters():
        param.requires_grad = False

    # Verify CLIP weights are loaded
    if args.local_rank == 0:
        logger.info("\n=== CLIP Weight Verification ===")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        clip_params = sum(p.numel() for p in model.clip.parameters())
        cross_params = sum(p.numel() for p in model.cross.parameters())

        logger.info(f"Total model parameters: {total_params:,}")
        logger.info(
            f"CLIP (visual + text encoders) parameters: {clip_params:,} ({100*clip_params/total_params:.1f}%)"
        )
        logger.info(
            f"Cross-attention parameters: {cross_params:,} ({100*cross_params/total_params:.1f}%)"
        )

        # Check for CLIP-specific components
        if hasattr(model.clip, "visual"):
            logger.info(f"Visual encoder loaded: {type(model.clip.visual).__name__}")
        if hasattr(model.clip, "transformer"):
            logger.info(
                f"Text encoder loaded: Transformer with {model.clip.transformer.layers} layers"
            )
        if hasattr(model.clip, "token_embedding"):
            logger.info(
                f"Token embedding loaded: vocabulary size {model.clip.token_embedding.num_embeddings}"
            )
        if hasattr(model.clip, "positional_embedding"):
            logger.info(
                f"Positional embedding loaded: shape {model.clip.positional_embedding.shape}"
            )
        if hasattr(model.clip, "logit_scale"):
            logger.info(
                f"Logit scale parameter loaded: {model.clip.logit_scale.item():.4f}"
            )

        # Verify trainable parameters in CLIP
        clip_trainable = sum(
            p.numel() for p in model.clip.parameters() if p.requires_grad
        )
        logger.info(
            f"\nCLIP trainable parameters: {clip_trainable:,} ({100*clip_trainable/clip_params:.1f}%)"
        )

        logger.info("=== END VERIFICATION ===\n")

    return model


def prep_optimizer(
    args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.0
):
    """
    Prepare optimizer with different learning rates for CLIP and cross-attention.

    Strategy:
    - CLIP layers: lr * coef_lr (slower, e.g., 1e-5 * 1e-3 = 1e-8)
    - Cross-attention layers: lr (faster, e.g., 1e-5)
    - This prevents catastrophic forgetting of pre-trained CLIP weights
    """
    global logger

    if hasattr(model, "module"):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # Separate decay and no-decay parameters
    decay_param_tp = [
        (n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)
    ]
    no_decay_param_tp = [
        (n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)
    ]

    # Separate CLIP and non-CLIP parameters
    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [
        (n, p) for n, p in no_decay_param_tp if "clip." not in n
    ]

    weight_decay = 0.2
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in decay_clip_param_tp],
            "weight_decay": weight_decay,
            "lr": args.lr * coef_lr,
        },
        {"params": [p for n, p in decay_noclip_param_tp], "weight_decay": weight_decay},
        {
            "params": [p for n, p in no_decay_clip_param_tp],
            "weight_decay": 0.0,
            "lr": args.lr * coef_lr,
        },
        {"params": [p for n, p in no_decay_noclip_param_tp], "weight_decay": 0.0},
    ]

    scheduler = None
    optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=args.lr,
        warmup=args.warmup_proportion,
        schedule="warmup_cosine",
        b1=0.9,
        b2=0.98,
        e=1e-6,
        t_total=num_train_optimization_steps,
        weight_decay=weight_decay,
        max_grad_norm=1.0,
    )

    if logger and args.local_rank == 0:
        logger.info(f"Optimizer initialized:")
        logger.info(f"  Learning rate: {args.lr}")
        logger.info(f"  CLIP learning rate: {args.lr * coef_lr}")
        logger.info(f"  Warmup proportion: {args.warmup_proportion}")
        logger.info(f"  Total optimization steps: {num_train_optimization_steps}")

        # Detailed optimizer configuration
        logger.info(f"\nOptimizer groups:")
        for i, group in enumerate(optimizer_grouped_parameters):
            group_lr = group.get("lr", args.lr)
            group_wd = group.get("weight_decay", 0)
            num_params = sum(p.numel() for p in group["params"])
            logger.info(
                f"  Group {i}: {num_params:,} params, lr={group_lr:.2e}, weight_decay={group_wd}"
            )

        logger.info(f"\n✓ CLIP layers will train at SLOW rate: {args.lr * coef_lr:.2e}")
        logger.info(f"✓ Cross-attention layers will train at FAST rate: {args.lr:.2e}")

    # NOTE: DDP wrapping is now done in main() BEFORE optimizer creation
    # This ensures the optimizer references the correct DDP-wrapped parameters
    # We do NOT wrap here to avoid double-wrapping

    return optimizer, scheduler, model


def save_model(epoch, args, model, optimizer, tr_loss, type_name=""):
    """Save model checkpoint."""
    global logger

    # Only save the model itself
    model_to_save = model.module if hasattr(model, "module") else model
    output_model_file = os.path.join(
        args.output_dir,
        f"pytorch_model.bin.{'' if type_name=='' else type_name+'.'}{epoch}",
    )
    optimizer_state_file = os.path.join(
        args.output_dir,
        f"pytorch_opt.bin.{'' if type_name=='' else type_name+'.'}{epoch}",
    )

    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save(
        {
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": tr_loss,
        },
        optimizer_state_file,
    )

    if logger:
        logger.info(f"Model saved to {output_model_file}")
        logger.info(f"Optimizer saved to {optimizer_state_file}")

    return output_model_file


def train_epoch(
    epoch,
    args,
    model,
    train_dataloader,
    val_dataloader,
    device,
    optimizer,
    global_step,
    tb_writer=None,
    debug=False,
):
    """
    Train for one epoch.

    Training loop:
    1. For each batch:
       a. Forward pass: model(text, video) -> loss
       b. Backward pass: loss.backward()
       c. Update weights: optimizer.step()
    2. Validate at end of epoch
    3. Save checkpoint

    Loss computation:
    - Contrastive loss between video and text embeddings
    - In-batch negatives: Other videos/texts in the same batch
    - Positive pairs: (video_i, text_i) from same data sample
    - Negative pairs: All other combinations
    """
    logger.info(f"Epoch {epoch + 1}/{args.epochs}")

    model.train()
    total_loss = 0
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    for step, batch in tqdm.tqdm(
        enumerate(train_dataloader), total=len(train_dataloader)
    ):
        start_time.record()

        # Unpack batch (last two items are lists of strings, not tensors)
        input_ids, input_mask, segment_ids, video, video_mask, video_ids, queries = (
            batch
        )

        # Debug output
        if debug and step % args.n_display == 0:
            logger.info(f"\n=== DEBUG INFO (Step {step}) ===")
            logger.info(f"Batch size: {len(video_ids)}")
            logger.info(f"Video frames shape: {video.shape}")
            logger.info(f"Text tokens shape: {input_ids.shape}")

            # Create debug frames directory if it doesn't exist
            debug_frames_dir = os.path.join(args.output_dir, "debug_frames")
            os.makedirs(debug_frames_dir, exist_ok=True)

            # Save frames with text overlay
            for idx in range(len(video_ids)):
                logger.info(
                    f"  [{idx}] Video ID: {video_ids[idx]}, Caption: {queries[idx]}"
                )

                # Extract frames for this sample
                # video shape: [batch, max_frames, 1, 3, H, W]
                sample_video = video[idx, :, 0, :, :, :]  # [max_frames, 3, H, W]

                # Save first, middle, and last frames
                num_frames = sample_video.shape[0]

                for frame_idx in range(num_frames):
                    frame = sample_video[frame_idx]  # [3, H, W]
                    save_frame_with_text(
                        frame, video_ids[idx], queries[idx], debug_frames_dir, frame_idx
                    )

            logger.info(f"Frames saved to {debug_frames_dir}")
            logger.info(f"=== END DEBUG ===")

        # Move batch to device
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        video = video.to(device)
        video_mask = video_mask.to(device)

        # Reshape video for model input
        # DataLoader stacks: [batch, 1, max_frames, 1, 3, H, W]
        # Model expects: [batch, pair, bs, ts, channel, h, w]
        # where pair=1, bs=1, ts=max_frames
        # So we need: [batch, 1, 1, max_frames, 3, H, W]

        # Current shape after DataLoader: [batch, 1, max_frames, 1, 3, H, W]
        # Need to rearrange to: [batch, 1, 1, max_frames, 3, H, W]
        b = video.shape[0]
        video = video.squeeze(3)  # Remove the singleton at dim=3
        # Now: [batch, 1, max_frames, 3, H, W]
        video = video.unsqueeze(2)  # Add bs=1 dimension
        # Now: [batch, 1, 1, max_frames, 3, H, W]

        # Forward pass
        # model() computes contrastive loss internally
        loss = model(input_ids, segment_ids, input_mask, video, video_mask)

        # Handle gradient accumulation
        # DDP automatically averages gradients across GPUs during backward pass
        # We only scale by gradient_accumulation_steps, not by world_size
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Accumulate true loss (not scaled) for logging
        total_loss += float(loss)

        # Update weights every gradient_accumulation_steps
        if (step + 1) % args.gradient_accumulation_steps == 0:
            # Clip gradients to prevent exploding gradients
            clip_grad_norm_(model.parameters(), 1.0)

            # Update weights
            optimizer.step()
            optimizer.zero_grad()

            # Clamp logit scale (CLIP-specific)
            # Prevents similarity scores from becoming too extreme
            if hasattr(model, "module"):
                torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            else:
                torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

            global_step += 1

            # Log progress
            if global_step % args.n_display == 0:
                end_time.record()
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # ms to s

                lr_str = "-".join(
                    [f"{lr:.9f}" for lr in sorted(list(set(optimizer.get_lr())))]
                )

                logger.info(
                    f"Epoch: {epoch + 1}/{args.epochs}, "
                    f"Step: {step + 1}/{len(train_dataloader)}, "
                    f"Lr: {lr_str}, "
                    f"Loss: {float(loss):.4f}, "
                    f"Time/step: {elapsed_time / args.n_display:.3f}s"
                )

                # Log to tensorboard (only from rank 0 to avoid duplicate logs)
                if tb_writer is not None and args.rank == 0:
                    tb_writer.add_scalar("train/loss", float(loss), global_step)
                    tb_writer.add_scalar("train/lr", optimizer.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "train/time_per_step",
                        elapsed_time / args.n_display,
                        global_step,
                    )

                start_time.record()

    avg_loss = total_loss / len(train_dataloader)

    # Synchronize average loss across all processes
    if args.world_size > 1:
        avg_loss_tensor = torch.tensor(avg_loss, device=device, dtype=torch.float32)
        torch.distributed.all_reduce(avg_loss_tensor, op=torch.distributed.ReduceOp.SUM)
        avg_loss = (avg_loss_tensor / args.world_size).item()

    if args.local_rank == 0:
        logger.info(f"\nEpoch {epoch + 1} finished. Average loss: {avg_loss:.4f}")

    # Log epoch average loss to tensorboard (only from rank 0)
    if tb_writer is not None and args.rank == 0:
        tb_writer.add_scalar("train/epoch_avg_loss", avg_loss, epoch)

    # Validation
    if val_dataloader is not None:
        val_loss = validate(args, model, val_dataloader, device)

        if args.local_rank == 0:
            logger.info(f"Validation loss: {val_loss:.4f}")

        # Log validation loss to tensorboard (only from rank 0)
        if tb_writer is not None and args.rank == 0:
            tb_writer.add_scalar("val/loss", val_loss, epoch)

    return avg_loss, global_step


def validate(args, model, val_dataloader, device):
    """
    Validate model on validation set.

    Computes average loss without updating weights.

    Note: Keep model in training mode so that forward() computes loss.
    In eval mode, CLIP4Clip forward() returns None (for inference-only).
    """
    logger.info("\nValidating...")

    # Keep model in training mode but disable gradients
    # This ensures forward() computes loss (self.training = True)
    # But doesn't update weights (torch.no_grad())
    model.train()

    total_loss = 0
    valid_batches = 0

    with torch.no_grad():
        for batch in val_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, video, video_mask = batch

            # Reshape video for model input
            # Current shape after DataLoader: [batch, 1, max_frames, 1, 3, H, W]
            # Need to rearrange to: [batch, 1, 1, max_frames, 3, H, W]
            video = video.squeeze(3)  # Remove singleton at dim=3
            video = video.unsqueeze(2)  # Add bs=1 dimension

            try:
                loss = model(input_ids, segment_ids, input_mask, video, video_mask)
                if loss is not None:
                    total_loss += float(loss)
                    valid_batches += 1
                else:
                    logger.warning(
                        "Validation batch returned None loss (should not happen in training mode)"
                    )
            except Exception as e:
                logger.warning(f"Validation batch failed: {e}")
                import traceback

                logger.warning(traceback.format_exc())
                continue

    if valid_batches > 0:
        avg_loss = total_loss / valid_batches
    else:
        logger.warning("No valid validation batches!")
        avg_loss = 0.0

    # Synchronize validation loss across all processes
    if args.world_size > 1:
        avg_loss_tensor = torch.tensor(avg_loss, device=device, dtype=torch.float32)
        torch.distributed.all_reduce(avg_loss_tensor, op=torch.distributed.ReduceOp.SUM)
        avg_loss = (avg_loss_tensor / args.world_size).item()

    # Keep in training mode for next epoch
    # (train_epoch will be called next)

    return avg_loss


def main():
    global logger
    global tb_writer

    # Parse arguments
    args = get_args()
    args = set_seed_logger(args)

    # Initialize tensorboard writer
    tb_log_dir = os.path.join(args.output_dir, "tensorboard")
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(tb_log_dir)

    # Get local_rank from args for distributed training
    local_rank = args.local_rank
    device, n_gpu = init_device(args, local_rank)

    # Initialize tokenizer
    tokenizer = ClipTokenizer()

    # Load dataset (only log from main process in distributed training)
    if local_rank == 0:
        logger.info("Loading MammAlps Dataset")

    # Load training dataset
    if local_rank == 0:
        logger.info(f"\nLoading training data from: {args.train_csv}")

    train_dataset = MammAlpsDataLoader(
        csv_path=args.train_csv,
        video_dir=args.train_video_dir,
        tokenizer=tokenizer,
        max_words=args.max_words,
        max_frames=args.max_frames,
        feature_framerate=args.feature_framerate,
        image_resolution=args.image_resolution,
        slice_framepos=2,
    )

    # Load test/validation dataset (if provided)
    val_dataset = None
    if args.test_csv and args.test_video_dir:
        if local_rank == 0:
            logger.info(f"Loading test data from: {args.test_csv}")

        val_dataset = MammAlpsDataLoader(
            csv_path=args.test_csv,
            video_dir=args.test_video_dir,
            tokenizer=tokenizer,
            max_words=args.max_words,
            max_frames=args.max_frames,
            feature_framerate=args.feature_framerate,
            image_resolution=args.image_resolution,
            slice_framepos=2,
        )

    if local_rank == 0:
        logger.info(f"\nDataset loaded:")
        logger.info(f"  Training samples: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"  Validation samples: {len(val_dataset)}")
        else:
            logger.info(f"  Validation: None (no test data provided)")

    # Create dataloaders with BalancedDistributedSampler for balanced multi-GPU data distribution
    if local_rank == 0:
        logger.info(f"\nDistributed Training Setup (with Balanced Sampling):")
        logger.info(f"  World size (total GPUs): {args.world_size}")
        logger.info(f"  Global rank: {args.rank}")
        logger.info(f"  Local rank: {args.local_rank}")

    train_sampler = BalancedDistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
        replacement=True,  # Allow sampling the same index multiple times to balance captions
    )

    train_dataloader = data_utils.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0,
        collate_fn=custom_collate_fn,
    )

    if local_rank == 0:
        logger.info(f"\nDataLoader Configuration:")
        logger.info(f"  Batch size per GPU: {args.batch_size}")
        logger.info(
            f"  Effective batch size (all GPUs): {args.batch_size * args.world_size}"
        )
        logger.info(f"  Training samples per GPU per epoch: {len(train_sampler)}")
        logger.info(
            f"  Total training samples processed per epoch: {len(train_sampler) * args.world_size}"
        )
        logger.info(
            f"  ✓ Balanced sampling: Videos with rare captions sampled more often"
        )

    val_dataloader = None
    if val_dataset is not None:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
            seed=args.seed,
            drop_last=False,
        )
        val_dataloader = data_utils.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=0,
            collate_fn=custom_collate_fn,
        )

    model = init_model(args, device, n_gpu, local_rank)
    print(model)

    # IMPORTANT: Wrap model in DDP BEFORE creating optimizer
    # This ensures optimizer references the DDP-wrapped parameters
    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        if local_rank == 0:
            logger.info(f"✓ Model wrapped in DistributedDataParallel")

    # Now create optimizer (after DDP wrapping)
    num_train_optimization_steps = (
        int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
        / args.gradient_accumulation_steps
    ) * args.epochs

    # prep_optimizer returns (optimizer, scheduler, model)
    optimizer, scheduler, model = prep_optimizer(
        args,
        model,
        num_train_optimization_steps,
        device,
        n_gpu,
        local_rank,
        coef_lr=args.coef_lr,
    )

    # Training loop
    # Barrier: wait for all processes after model/optimizer setup
    if args.world_size > 1:
        torch.distributed.barrier()

    if local_rank == 0:
        logger.info(f"\nTraining configuration:")
        logger.info(f"  Epochs: {args.epochs}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(
            f"  Effective batch size (all GPUs): {args.batch_size * args.world_size}"
        )
        logger.info(f"  Steps per epoch: {len(train_dataloader)}")
        logger.info(f"  Total optimization steps: {num_train_optimization_steps}")
        logger.info(f"  Local rank: {local_rank}")
        logger.info(f"  World size: {args.world_size}")
        logger.info(f"  DDP enabled: {args.world_size > 1}")

    global_step = 0
    for epoch in tqdm.tqdm(range(args.epochs), desc="Training Epochs"):
        # Set epoch for sampler (ensures different data order each epoch in distributed mode)
        train_sampler.set_epoch(epoch)
        if val_dataloader is not None and hasattr(val_dataloader.sampler, "set_epoch"):
            val_dataloader.sampler.set_epoch(epoch)

        train_loss, global_step = train_epoch(
            epoch,
            args,
            model,
            train_dataloader,
            val_dataloader,
            device,
            optimizer,
            global_step,
            tb_writer,
            debug=args.debug,
        )

        # Synchronize all processes to ensure checkpoint is written before next epoch
        if args.world_size > 1:
            torch.distributed.barrier()

        # Save checkpoint after each epoch (only from main process)
        if local_rank == 0:
            save_model(epoch, args, model, optimizer, train_loss)

    # Close tensorboard writer
    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    main()
