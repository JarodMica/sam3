import glob
import os
import subprocess
import sys
import uuid
from typing import Dict, List, Optional, Tuple
import contextlib
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import gradio as gr
import numpy as np
import torch
from collections import defaultdict
from PIL import Image, ImageDraw
from pathlib import Path

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import COLORS, prepare_masks_for_visualization, render_masklet_frame


def _bf16_autocast_if_cuda():
    """Use bf16 autocast on CUDA when available; no-op otherwise."""
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()

# Lazily instantiated globals (created via the "Load models" button)
IMAGE_MODEL = None
IMAGE_PROCESSOR: Optional[Sam3Processor] = None
VIDEO_PREDICTOR = None
TEMP_IMAGE_PATH = Path(__file__).resolve().parent / "temp.jpg"
GRADIO_TEMP = Path(__file__).resolve().parent / "temp_videos" / "gradio_temp"
TEMP_VIDEO_ROOT = Path(__file__).resolve().parent / "temp_video"

# Keep Gradio uploads out of the default OS temp (e.g., AppData) by redirecting its temp dir.
os.environ.setdefault("GRADIO_TEMP_DIR", str(GRADIO_TEMP))
GRADIO_TEMP.mkdir(parents=True, exist_ok=True)
TEMP_VIDEO_ROOT.mkdir(parents=True, exist_ok=True)

# ------------------------------
# Utility helpers
# ------------------------------
def _np_image(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def _color_for_idx(idx: int) -> Tuple[int, int, int]:
    color = (COLORS[idx % len(COLORS)] * 255).astype(np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


def _mask_to_crop(img: Image.Image, mask: np.ndarray) -> Optional[Image.Image]:
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return None
    y_idx, x_idx = np.where(mask_bool)
    x0, x1 = x_idx.min(), x_idx.max() + 1
    y0, y1 = y_idx.min(), y_idx.max() + 1
    img_np = _np_image(img)
    crop_np = img_np[y0:y1, x0:x1]
    mask_crop = mask_bool[y0:y1, x0:x1]
    crop_np = np.where(mask_crop[..., None], crop_np, 0)
    return Image.fromarray(crop_np)


def _draw_star(draw: ImageDraw.Draw, x: float, y: float, size: int, fill: Tuple[int, int, int]):
    # draw a simple 5-point star
    cx, cy = x, y
    r = size
    points = []
    for i in range(10):
        angle = np.pi / 2 + i * np.pi / 5
        rad = r if i % 2 == 0 else r / 2
        points.append((cx + rad * np.cos(angle), cy - rad * np.sin(angle)))
    draw.polygon(points, fill=fill, outline=(255, 255, 255))


def _overlay_points(base_img: Image.Image, points: List[Dict]) -> Image.Image:
    img = _np_image(base_img)
    overlay = Image.fromarray(img)
    draw = ImageDraw.Draw(overlay)
    for p in points:
        x, y = p["x"], p["y"]
        color = (0, 200, 0) if p["label"] == 1 else (200, 0, 0)
        _draw_star(draw, x, y, size=12, fill=color)
    return overlay


def _save_temp_image(img: Image.Image):
    TEMP_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(TEMP_IMAGE_PATH)


def _load_temp_image() -> Optional[Image.Image]:
    if TEMP_IMAGE_PATH.exists():
        return Image.open(TEMP_IMAGE_PATH).convert("RGB")
    return None


def _points_to_table(points_state: List[Dict]) -> List[List]:
    rows = []
    for p in points_state:
        label = "Positive" if p["label"] == 1 else "Negative"
        x = int(p["x"])
        y = int(p["y"])
        obj_id = int(p.get("obj_id", 0))
        rows.append([label, x, y, obj_id])
    return rows


def _table_to_points(table_data) -> List[Dict]:
    points = []
    if table_data is None:
        return points
    for row in table_data:
        if row is None or len(row) < 4:
            continue
        label_str, x, y, obj_id = row
        if label_str not in ("Positive", "Negative"):
            continue
        try:
            x_val = float(x)
            y_val = float(y)
            obj_val = int(obj_id)
        except (TypeError, ValueError):
            continue
        points.append(
            {
                "label": 1 if label_str == "Positive" else 0,
                "x": x_val,
                "y": y_val,
                "obj_id": obj_val,
            }
        )
    return points


def _overlay_masks(
    base_img: Image.Image,
    masks: List[np.ndarray],
    boxes: Optional[List[np.ndarray]] = None,
    points: Optional[List[Dict]] = None,
) -> Image.Image:
    """Blend masks/boxes/points over the image for visualization."""
    img_np = _np_image(base_img)
    overlay = img_np.copy()

    for idx, mask in enumerate(masks):
        mask_bool = mask.astype(bool)
        if mask_bool.ndim == 3:
            mask_bool = mask_bool.squeeze(0)
        if mask_bool.shape[:2] != overlay.shape[:2]:
            continue
        color = _color_for_idx(idx)
        for c in range(3):
            overlay[..., c][mask_bool] = (
                0.6 * color[c] + 0.4 * overlay[..., c][mask_bool]
            ).astype(np.uint8)

    draw = ImageDraw.Draw(Image.fromarray(overlay))
    if boxes:
        for idx, box in enumerate(boxes):
            color = _color_for_idx(idx)
            x0, y0, x1, y1 = [float(v) for v in box]
            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
    if points:
        for pt in points:
            x, y, label = pt["x"], pt["y"], pt["label"]
            color = (0, 200, 0) if label == 1 else (200, 0, 0)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline="white")
    return Image.fromarray(overlay)


def _load_video_frame(video_state: Dict, frame_idx: int) -> Image.Image:
    video_path = video_state["video_path"]
    if os.path.isdir(video_path):
        frame_paths = video_state["frame_paths"]
        frame_idx = max(0, min(frame_idx, len(frame_paths) - 1))
        frame_path = frame_paths[frame_idx]
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        cap.release()
        if not success:
            raise RuntimeError(f"Unable to read frame {frame_idx}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def _gather_masks_from_text_state(state: Dict) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    masks = []
    boxes = []
    scores: List[float] = []
    if "masks" in state and state["masks"] is not None:
        for m in state["masks"]:
            masks.append(m.squeeze().cpu().numpy())
    if "boxes" in state and state["boxes"] is not None:
        for b in state["boxes"]:
            boxes.append(b.cpu().numpy())
    if "scores" in state and state["scores"] is not None:
        for s in state["scores"]:
            scores.append(float(s.item()))
    return masks, boxes, scores


# ------------------------------
# Model loading
# ------------------------------
def load_models(progress=gr.Progress(track_tqdm=True)):
    """Load/download the SAM3 image + video models."""
    global IMAGE_MODEL, IMAGE_PROCESSOR, VIDEO_PREDICTOR
    logs = []

    # To avoid bf16 conv bias mismatches in this app, default to full precision for the tracker unless the user overrides.
    os.environ.setdefault("SAM3_DISABLE_BF16", "1")

    if IMAGE_MODEL is None:
        progress(0.1, desc="Loading image model")
        IMAGE_MODEL = build_sam3_image_model(enable_inst_interactivity=True)
        IMAGE_PROCESSOR = Sam3Processor(IMAGE_MODEL, confidence_threshold=0.5)
        device = next(IMAGE_MODEL.parameters()).device
        logs.append(f"Image model ready on {device}.")
    else:
        logs.append("Image model already loaded.")

    progress(1.0, desc="Done")
    return "\n".join(logs)


# ------------------------------
# Image tab callbacks
# ------------------------------
def set_image_session(image: Image.Image):
    if IMAGE_PROCESSOR is None:
        return None, [], [], [], None, None, "Load models before uploading an image."
    if image is None:
        return None, [], [], [], None, None, "Please upload an image."

    _save_temp_image(image)
    state = IMAGE_PROCESSOR.set_image(image, {})
    session = {"state": state, "orig_image": image.copy(), "image": image.copy()}
    return (
        session,
        [],
        [],
        [],
        image,
        image,
        "Image embedded. Click to add points, then Render to see masks.",
    )


def handle_image_click(
    display_image: Image.Image,
    point_label: str,
    point_obj_id: float,
    points_state: List[Dict],
    image_session: Optional[Dict],
    evt: gr.SelectData,
):
    """Add a point and show point markers only (no masks)."""
    base = _load_temp_image() or (image_session.get("orig_image") if image_session else None)
    if base is None:
        return display_image, points_state, gr.update(), gr.update(), gr.update(), "Upload an image first."

    if evt is None or evt.index is None:
        return display_image, points_state, gr.update(), gr.update(), gr.update(), "Click data missing."

    x, y = evt.index
    label = 1 if point_label == "Positive" else 0
    obj_id = int(point_obj_id) if point_obj_id is not None else 0
    new_points = list(points_state) + [{"x": x, "y": y, "label": label, "obj_id": obj_id}]
    overlay = _overlay_points(base, new_points)
    table = _points_to_table(new_points)
    return overlay, new_points, gr.update(), gr.update(), table, f"Added {point_label.lower()} point (obj {obj_id}) at ({int(x)}, {int(y)})."


def clear_image_points(image_session: Optional[Dict]):
    if image_session is None or image_session.get("orig_image") is None:
        return None, [], gr.update(), gr.update(), [], "Nothing to clear."
    base = _load_temp_image() or image_session.get("orig_image", None)
    return base, [], base, gr.update(), [], "Cleared clicks."


def reset_image_view(image_session: Optional[Dict]):
    """Restore the original uploaded image and drop all points/masks."""
    if image_session is None or image_session.get("orig_image") is None:
        return None, [], gr.update(), gr.update(), [], "Nothing to reset."
    base = _load_temp_image() or image_session.get("orig_image", None)
    return base, [], base, gr.update(), [], "Restored original image."


def run_image_inference(
    text_prompt: str,
    multimask: bool,
    image_session: Optional[Dict],
    points_state: List[Dict],
    current_image: Optional[Image.Image] = None,
):
    if IMAGE_MODEL is None or IMAGE_PROCESSOR is None:
        raise gr.Error("Load models before running inference.")
    if image_session is None or image_session.get("orig_image") is None:
        raise gr.Error("Upload an image first.")

    img = _load_temp_image() or image_session.get("orig_image", None)
    if img is None:
        raise gr.Error("Original image missing; please re-upload.")
    state = image_session["state"]
    IMAGE_PROCESSOR.reset_all_prompts(state)
    masks: List[np.ndarray] = []
    boxes: List[np.ndarray] = []
    logs = []

    if text_prompt and text_prompt.strip():
        state = IMAGE_PROCESSOR.set_text_prompt(prompt=text_prompt.strip(), state=state)
        text_masks, text_boxes, scores = _gather_masks_from_text_state(state)
        masks.extend(text_masks)
        boxes.extend(text_boxes)
        logs.append(f"Text prompt found {len(text_masks)} mask(s) with scores {scores}.")

    if points_state:
        groups = defaultdict(list)
        for p in points_state:
            groups[p.get("obj_id", 0)].append(p)
        for obj_id, pts in sorted(groups.items(), key=lambda x: x[0]):
            point_coords = np.array([[p["x"], p["y"]] for p in pts], dtype=np.float32)
            point_labels = np.array([p["label"] for p in pts], dtype=np.int32)
            masks_np, ious_np, _ = IMAGE_MODEL.predict_inst(
                inference_state=state,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=multimask,
                normalize_coords=True,
            )
            best_idx = int(np.argmax(ious_np)) if len(ious_np) else 0
            if masks_np.ndim == 3:
                masks.append(masks_np[best_idx])
            logs.append(
                f"Obj {obj_id}: point prompt produced {masks_np.shape[0]} mask(s); showing best idx {best_idx} with IoU {float(ious_np[best_idx]) if len(ious_np) else 'n/a'}."
            )

    if not masks:
        raise gr.Error("No masks produced. Try another prompt.")

    overlay = _overlay_masks(img, masks=masks, boxes=boxes, points=points_state)

    crops: List[Image.Image] = []
    for m in masks:
        crop = _mask_to_crop(img, m)
        if crop is not None:
            crops.append(crop)

    return overlay, crops, _points_to_table(points_state), "\n".join(logs)


def apply_points_table(table_data, text_prompt, multimask, image_session, current_image):
    """Re-run segmentation from edited table rows."""
    points = _table_to_points(table_data)
    mask_overlay, crops, table, log = run_image_inference(
        text_prompt=text_prompt,
        multimask=multimask,
        image_session=image_session,
        points_state=points,
        current_image=current_image,
    )
    base = _load_temp_image() or (image_session.get("orig_image") if image_session else None)
    clickable_overlay = _overlay_points(base, points) if base is not None else current_image
    return clickable_overlay, mask_overlay, points, crops, table, f"Re-applied table edits.\n{log}"


# ------------------------------
# Video tab callbacks (chunked subprocess pipeline)
# ------------------------------
def _parse_video_dir(video_path: str) -> List[str]:
    frame_paths = glob.glob(os.path.join(video_path, "*.jpg"))
    try:
        frame_paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    except ValueError:
        frame_paths.sort()
    return frame_paths


def _extract_video_to_frames(video_path: str, dest_root: Path) -> List[str]:
    """Extract frames with ffmpeg to a temp directory and return sorted paths."""
    dest_root.mkdir(parents=True, exist_ok=True)
    out_pattern = dest_root / "%06d.jpg"
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-q:v",
        "2",
        "-y",
        str(out_pattern),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return _parse_video_dir(str(dest_root))


def _copy_video_to_temp(video_file: str) -> Path:
    """Copy the uploaded video into the app temp_video folder (replacing if exists)."""
    src = Path(video_file)
    dest = TEMP_VIDEO_ROOT / src.name
    if dest.exists():
        if dest.is_file():
            dest.unlink()
        else:
            shutil.rmtree(dest)
    shutil.copy2(src, dest)
    return dest


def _read_video_metadata(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    fps_val = cap.get(cv2.CAP_PROP_FPS)
    frame_count_val = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    success, first_frame = cap.read()
    cap.release()
    if not success or first_frame is None:
        raise RuntimeError("Unable to read video.")
    fps = float(fps_val) if fps_val and fps_val > 0 else 10.0
    frame_count = int(frame_count_val) if frame_count_val and frame_count_val > 0 else 1
    sample_frame = Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
    return fps, frame_count, sample_frame


def prepare_video_upload_chunked(video_file: str, prev_state: Optional[Dict]):
    """Copy upload to temp_video, return preview + state."""
    if video_file is None:
        return None, None, None, "Please upload a video file."
    try:
        dest = _copy_video_to_temp(video_file)
        fps, frame_count, sample_frame = _read_video_metadata(dest)
    except Exception as exc:
        return None, None, None, f"Failed to load video: {exc}"

    state = {
        "video_path": str(dest),
        "frame_count": frame_count,
        "width": sample_frame.width,
        "height": sample_frame.height,
        "fps": fps,
    }
    return sample_frame, state, f"Video copied to temp_video and ready. Frames: {frame_count}, FPS: {fps:.2f}"


def _chunk_frames(frames_dir: Path, chunk_size: int, chunks_root: Path) -> List[Path]:
    frames = _parse_video_dir(str(frames_dir))
    if not frames:
        raise RuntimeError("No frames extracted from video.")
    chunks_root.mkdir(parents=True, exist_ok=True)
    chunk_dirs: List[Path] = []
    for start in range(0, len(frames), chunk_size):
        chunk_idx = len(chunk_dirs)
        chunk_dir = chunks_root / f"chunk_{chunk_idx:04d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for frame_path in frames[start : start + chunk_size]:
            src = Path(frame_path)
            dest = chunk_dir / src.name
            shutil.move(str(src), dest)
        chunk_dirs.append(chunk_dir)
    return chunk_dirs


def _run_chunk_subprocess(chunk_dir: Path, out_dir: Path, prompt_text: str, checkpoint: str) -> Tuple[Path, int, str, str]:
    worker_path = Path(__file__).resolve().parent / "video_chunk_worker.py"
    cmd = [
        sys.executable,
        str(worker_path),
        "--frames-dir",
        str(chunk_dir),
        "--output-dir",
        str(out_dir),
        "--prompt-text",
        prompt_text or "",
        "--checkpoint",
        checkpoint,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return out_dir, result.returncode, result.stdout, result.stderr


def _stitch_outputs(output_root: Path, fps: float, dest_path: Path):
    frame_files = list(output_root.rglob("*.jpg"))
    if not frame_files:
        raise RuntimeError("No rendered frames to stitch.")
    try:
        frame_files.sort(key=lambda p: int(p.stem))
    except ValueError:
        frame_files.sort()
    sample = cv2.imread(str(frame_files[0]))
    height, width, _ = sample.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dest_path), fourcc, fps, (width, height))
    for fp in frame_files:
        img = cv2.imread(str(fp))
        writer.write(img)
    writer.release()
    return dest_path


def run_chunked_video_processing(
    text_prompt: str,
    chunk_size: int,
    parallel_chunks: int,
    fps_override: float,
    video_state: Optional[Dict],
):
    if video_state is None or not video_state.get("video_path"):
        raise gr.Error("Upload a video first.")
    if chunk_size <= 0:
        raise gr.Error("Chunk size must be > 0.")
    video_path = Path(video_state["video_path"])
    job_dir = TEMP_VIDEO_ROOT / f"job_{uuid.uuid4().hex}"
    frames_dir = job_dir / "frames"
    chunks_root = job_dir / "chunks"
    outputs_root = job_dir / "outputs"
    outputs_root.mkdir(parents=True, exist_ok=True)

    logs = []
    logs.append(f"Job dir: {job_dir}")
    logs.append("Extracting frames...")
    frame_paths = _extract_video_to_frames(str(video_path), frames_dir)
    logs.append(f"Extracted {len(frame_paths)} frame(s).")

    logs.append(f"Chunking into size {chunk_size}...")
    chunk_dirs = _chunk_frames(frames_dir, chunk_size, chunks_root)
    logs.append(f"Created {len(chunk_dirs)} chunk(s).")

    prompt_text = (text_prompt or "").strip()
    checkpoint = str(
        Path(__file__).resolve().parent
        / "models/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
    )
    max_workers = max(1, min(parallel_chunks, len(chunk_dirs)))
    logs.append(f"Running up to {max_workers} chunk worker(s) in parallel...")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for chunk_dir in chunk_dirs:
            out_dir = outputs_root / chunk_dir.name
            futures.append(ex.submit(_run_chunk_subprocess, chunk_dir, out_dir, prompt_text, checkpoint))
        for f in as_completed(futures):
            out_dir, returncode, stdout, stderr = f.result()
            if returncode != 0:
                err_snip = (stderr or stdout or "").strip().splitlines()[-5:]
                raise gr.Error(
                    f"Chunk {out_dir.name} failed (return code {returncode}). "
                    f"Stderr tail:\n" + "\n".join(err_snip)
                )

    fps = fps_override if fps_override and fps_override > 0 else video_state.get("fps", 10.0)
    final_path = job_dir / "masked_video.mp4"
    logs.append(f"Stitching outputs at {fps:.2f} FPS...")
    stitched = _stitch_outputs(outputs_root, fps=fps, dest_path=final_path)
    logs.append(f"Done. Final video at {stitched}.")
    return str(stitched), "\n".join(logs)


# ------------------------------
# Gradio UI assembly
# ------------------------------
with gr.Blocks(title="SAM3 Gradio Demo") as demo:
    status_box = gr.Textbox(label="Logs", lines=6, interactive=False)
    with gr.Tabs():
        with gr.Tab("Image"):
            load_btn = gr.Button("Load models", variant="primary")
            load_btn.click(load_models, outputs=status_box)
            with gr.Row():
                with gr.Column(scale=1):
                    text_prompt = gr.Textbox(label="Text prompt", placeholder="e.g., cat, person, flower")
                    multimask_checkbox = gr.Checkbox(
                        value=True, label="Return multi-mask for clicks (take best automatically)"
                    )
                    point_label_radio = gr.Radio(
                        choices=["Positive", "Negative"], value="Positive", label="Click type"
                    )
                    point_obj_id_input = gr.Number(value=0, precision=0, label="Object ID for clicks", step=1)
                    clear_points_btn = gr.Button("Clear clicks")
                    reset_view_btn = gr.Button("Clear masking / Reset view")
                    run_image_btn = gr.Button("Run image segmentation", variant="primary")
                with gr.Column(scale=2):
                    image_clickable = gr.Image(label="Upload image (click to add points)", type="pil", interactive=True)
                    mask_output = gr.Image(label="Rendered masks", type="pil", interactive=False)
                    crop_gallery = gr.Gallery(label="Cropped regions", columns=3, height=200)
                with gr.Column(scale=1):
                    points_table = gr.Dataframe(
                        headers=["Label", "X", "Y", "Obj ID"],
                        datatype=["str", "number", "number", "number"],
                        row_count=(0, "dynamic"),
                        col_count=4,
                        label="Clicked points",
                        interactive=True,
                    )
                    apply_table_btn = gr.Button("Apply table edits")

            image_session_state = gr.State()
            image_points_state = gr.State([])

            image_clickable.upload(
                set_image_session,
                inputs=image_clickable,
                outputs=[image_session_state, image_points_state, crop_gallery, points_table, image_clickable, mask_output, status_box],
            )
            image_clickable.select(
                handle_image_click,
                inputs=[
                    image_clickable,
                    point_label_radio,
                    point_obj_id_input,
                    image_points_state,
                    image_session_state,
                ],
                outputs=[image_clickable, image_points_state, mask_output, crop_gallery, points_table, status_box],
            )
            clear_points_btn.click(
                clear_image_points,
                inputs=image_session_state,
                outputs=[image_clickable, image_points_state, mask_output, crop_gallery, points_table, status_box],
            )
            reset_view_btn.click(
                reset_image_view,
                inputs=image_session_state,
                outputs=[image_clickable, image_points_state, mask_output, crop_gallery, points_table, status_box],
            )
            apply_table_btn.click(
                apply_points_table,
                inputs=[points_table, text_prompt, multimask_checkbox, image_session_state, image_clickable],
                outputs=[image_clickable, mask_output, image_points_state, crop_gallery, points_table, status_box],
            )
            run_image_btn.click(
                run_image_inference,
                inputs=[text_prompt, multimask_checkbox, image_session_state, image_points_state, image_clickable],
                outputs=[mask_output, crop_gallery, points_table, status_box],
            )

        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Upload MP4 (saved to temp_video)")
                    text_prompt_video = gr.Textbox(label="Text prompt", placeholder="e.g., person", value="")
                    chunk_size_slider = gr.Slider(
                        value=500, minimum=1, maximum=1000, step=1, label="Chunk size (frames per subprocess)"
                    )
                    parallel_slider = gr.Slider(
                        value=1, minimum=1, maximum=4, step=1, label="Parallel chunks (subprocesses)"
                    )
                    fps_override_slider = gr.Slider(
                        value=0,
                        minimum=0,
                        maximum=120,
                        step=1,
                        label="Output FPS (0 = keep source)",
                    )
                    process_video_btn = gr.Button("Inference + propagate (chunked subprocess)", variant="primary")
                with gr.Column(scale=2):
                    video_preview = gr.Image(label="Preview frame", type="pil", interactive=False)
                    masked_video = gr.Video(label="Masked video", interactive=False)

            video_state = gr.State()

            video_input.change(
                prepare_video_upload_chunked,
                inputs=[video_input, video_state],
                outputs=[video_preview, video_state, status_box],
            )
            process_video_btn.click(
                run_chunked_video_processing,
                inputs=[text_prompt_video, chunk_size_slider, parallel_slider, fps_override_slider, video_state],
                outputs=[masked_video, status_box],
            )

if __name__ == "__main__":
    demo.launch()
