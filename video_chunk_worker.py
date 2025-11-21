"""
Lightweight subprocess worker to run SAM3 video inference on a folder of frames.

Usage:
    python video_chunk_worker.py --frames-dir /path/to/chunk --output-dir /path/to/out --prompt-text "person"

Each worker loads the video predictor from scratch (isolated CUDA context), runs
a text prompt on frame 0, propagates across all frames in the folder, renders
masked frames, and writes them to the output directory using the original frame
filenames.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import render_masklet_frame


def _sorted_frame_paths(frames_dir: Path):
    frames = list(frames_dir.glob("*.jpg"))
    try:
        frames.sort(key=lambda p: int(p.stem))
    except ValueError:
        frames.sort()
    return frames


def run_worker(frames_dir: Path, output_dir: Path, prompt_text: str, checkpoint_path: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = _sorted_frame_paths(frames_dir)
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}")

    predictor = build_sam3_video_predictor(checkpoint_path=checkpoint_path)
    resp = predictor.handle_request({"type": "start_session", "resource_path": str(frames_dir)})
    session_id = resp["session_id"]

    try:
        if prompt_text:
            predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": prompt_text,
                }
            )

        collected = {}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for response in predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "start_frame_index": 0,
                    "max_frame_num_to_track": None,
                    "propagation_direction": "forward",
                }
            ):
                collected[response["frame_index"]] = response["outputs"]

        for idx, outputs in sorted(collected.items()):
            if idx >= len(frame_paths):
                continue
            frame_path = frame_paths[idx]
            frame = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)
            rendered = render_masklet_frame(frame, outputs, frame_idx=idx, alpha=0.6)
            rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / frame_path.name), rendered_bgr)

        manifest = {"frames": [p.name for p in frame_paths], "outputs": sorted(collected.keys())}
        with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
        predictor.shutdown()


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAM3 chunk inference on frames in a directory.")
    parser.add_argument("--frames-dir", required=True, help="Directory of JPEG frames for this chunk.")
    parser.add_argument("--output-dir", required=True, help="Directory to write masked frames.")
    parser.add_argument(
        "--prompt-text",
        type=str,
        default="",
        help="Text prompt applied on frame 0 before propagation. Empty = no prompt.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt",
        help="Path to sam3.pt checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(__file__).resolve().parent / ckpt_path
    run_worker(frames_dir=frames_dir, output_dir=output_dir, prompt_text=args.prompt_text, checkpoint_path=str(ckpt_path))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[worker] failed: {exc}", file=sys.stderr)
        sys.exit(1)
