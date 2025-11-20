import glob
import os
from typing import Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
from collections import defaultdict
from PIL import Image, ImageDraw
from pathlib import Path

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import COLORS, prepare_masks_for_visualization, render_masklet_frame

# Lazily instantiated globals (created via the "Load models" button)
IMAGE_MODEL = None
IMAGE_PROCESSOR: Optional[Sam3Processor] = None
VIDEO_PREDICTOR = None
TEMP_IMAGE_PATH = Path(__file__).resolve().parent / "temp.jpg"


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

    if IMAGE_MODEL is None:
        progress(0.1, desc="Loading image model")
        IMAGE_MODEL = build_sam3_image_model(enable_inst_interactivity=True)
        IMAGE_PROCESSOR = Sam3Processor(IMAGE_MODEL, confidence_threshold=0.5)
        device = next(IMAGE_MODEL.parameters()).device
        logs.append(f"Image model ready on {device}.")
    else:
        logs.append("Image model already loaded.")

    if VIDEO_PREDICTOR is None:
        progress(0.6, desc="Loading video predictor")
        VIDEO_PREDICTOR = build_sam3_video_predictor()
        logs.append("Video predictor ready.")
    else:
        logs.append("Video predictor already loaded.")

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
# Video tab callbacks
# ------------------------------
def _parse_video_dir(video_path: str) -> List[str]:
    frame_paths = glob.glob(os.path.join(video_path, "*.jpg"))
    try:
        frame_paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    except ValueError:
        frame_paths.sort()
    return frame_paths


def start_video_session(video_file: str, prev_state: Optional[Dict]):
    if VIDEO_PREDICTOR is None:
        return None, None, None, None, "Load models before uploading a video."
    if video_file is None:
        return None, None, None, None, "Please upload a video file or folder."

    # Close any previous session
    if prev_state and prev_state.get("session_id"):
        try:
            VIDEO_PREDICTOR.handle_request({"type": "close_session", "session_id": prev_state["session_id"]})
        except Exception:
            pass

    video_path = video_file
    frame_paths = _parse_video_dir(video_path) if os.path.isdir(video_path) else None

    if frame_paths:
        frame_count = len(frame_paths)
        sample_frame = Image.open(frame_paths[0])
    else:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        success, frame = cap.read()
        cap.release()
        if not success:
            return None, None, None, None, [], "Could not read the first frame."
        sample_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    response = VIDEO_PREDICTOR.handle_request({"type": "start_session", "resource_path": video_path})
    session_id = response["session_id"]
    session_state = {
        "session_id": session_id,
        "video_path": video_path,
        "frame_count": frame_count,
        "frame_paths": frame_paths,
        "width": sample_frame.width,
        "height": sample_frame.height,
    }

    slider_update = gr.Slider.update(max=max(frame_count - 1, 0), value=0, interactive=True)
    return sample_frame, slider_update, session_state, [], f"Video loaded with {frame_count} frame(s). Session: {session_id}"


def refresh_video_frame(frame_idx: int, video_state: Optional[Dict], points_state: List[Dict]):
    if video_state is None:
        return None, "Load a video first."
    frame_img = _load_video_frame(video_state, frame_idx)
    points_on_frame = [p for p in points_state if p["frame"] == frame_idx]
    annotated = _overlay_masks(frame_img, masks=[], points=points_on_frame)
    return annotated, f"Showing frame {frame_idx}."


def handle_video_click(
    frame_image: Image.Image,
    point_label: str,
    frame_idx: int,
    points_state: List[Dict],
    video_state: Optional[Dict],
    evt: gr.SelectData,
):
    if video_state is None:
        return frame_image, points_state, "Load a video first."
    if evt is None or evt.index is None:
        return frame_image, points_state, "Click data missing."
    x, y = evt.index
    label = 1 if point_label == "Positive" else 0
    new_points = list(points_state) + [{"frame": frame_idx, "x": x, "y": y, "label": label}]
    points_on_frame = [p for p in new_points if p["frame"] == frame_idx]
    overlay = _overlay_masks(frame_image, masks=[], points=points_on_frame)
    return overlay, new_points, f"Added {point_label.lower()} point on frame {frame_idx} at ({int(x)}, {int(y)})."


def clear_video_points(frame_idx: int, video_state: Optional[Dict]):
    if video_state is None:
        return None, [], "Nothing to clear."
    frame_img = _load_video_frame(video_state, frame_idx)
    return frame_img, [], "Cleared clicks."


def run_video_inference(
    frame_idx: int,
    text_prompt: str,
    run_propagation: bool,
    max_frames: int,
    preview_stride: int,
    video_state: Optional[Dict],
    points_state: List[Dict],
):
    if VIDEO_PREDICTOR is None:
        return None, [], "Load models before running video inference."
    if video_state is None:
        return None, [], "Upload a video first."

    session_id = video_state["session_id"]
    VIDEO_PREDICTOR.handle_request({"type": "reset_session", "session_id": session_id})

    points_for_frame = [p for p in points_state if p["frame"] == frame_idx]
    pt_coords = None
    pt_labels = None
    if points_for_frame:
        pt_coords = torch.tensor(
            [[p["x"] / video_state["width"], p["y"] / video_state["height"]] for p in points_for_frame],
            dtype=torch.float32,
        )
        pt_labels = torch.tensor([p["label"] for p in points_for_frame], dtype=torch.int32)

    request = {
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": frame_idx,
        "text": text_prompt if text_prompt.strip() else None,
        "points": pt_coords,
        "point_labels": pt_labels,
    }
    response = VIDEO_PREDICTOR.handle_request(request)
    out = response["outputs"]
    frame_img = np.array(_load_video_frame(video_state, frame_idx))
    overlay = render_masklet_frame(frame_img, out, frame_idx=frame_idx, alpha=0.6)

    gallery = []
    log_lines = [f"Prompt applied on frame {frame_idx}. Objects: {len(out['out_obj_ids'])}."]

    if run_propagation:
        collected = {}
        for resp in VIDEO_PREDICTOR.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "start_frame_index": frame_idx,
                "max_frame_num_to_track": max_frames if max_frames > 0 else None,
            }
        ):
            collected[resp["frame_index"]] = resp["outputs"]
        collected = prepare_masks_for_visualization(collected)
        stride = max(1, preview_stride)
        selected_frames = sorted(collected.keys())[::stride]
        for fidx in selected_frames:
            frame_np = np.array(_load_video_frame(video_state, fidx))
            vis = render_masklet_frame(frame_np, collected[fidx], frame_idx=fidx, alpha=0.6)
            gallery.append((vis, f"Frame {fidx}"))
        log_lines.append(f"Propagation preview generated for {len(selected_frames)} frame(s) (stride={stride}).")

    return overlay, gallery, "\n".join(log_lines)


# ------------------------------
# Gradio UI assembly
# ------------------------------
with gr.Blocks(title="SAM3 Gradio Demo") as demo:
    status_box = gr.Textbox(label="Logs", lines=6, interactive=False)
    load_btn = gr.Button("Load models", variant="primary")
    load_btn.click(load_models, outputs=status_box)

    with gr.Tabs():
        with gr.Tab("Image"):
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
            video_input = gr.Video(label="Upload MP4 or folder path (mounted) for frames")
            frame_viewer = gr.Image(label="Selected frame (click to add points)", type="pil", interactive=True)
            frame_slider = gr.Slider(label="Frame index", minimum=0, maximum=0, value=0, step=1)
            point_label_video = gr.Radio(
                choices=["Positive", "Negative"], value="Positive", label="Click type"
            )
            with gr.Row():
                refresh_frame_btn = gr.Button("Refresh frame")
                clear_video_points_btn = gr.Button("Clear clicks")
            text_prompt_video = gr.Textbox(label="Text prompt (optional)", placeholder="e.g., person")
            run_propagation_chk = gr.Checkbox(value=False, label="Run propagation preview")
            max_frames_slider = gr.Slider(value=120, minimum=0, maximum=500, step=1, label="Max frames to track (0 = all)")
            stride_slider = gr.Slider(value=10, minimum=1, maximum=60, step=1, label="Preview stride")
            run_video_btn = gr.Button("Run video segmentation", variant="primary")
            video_overlay = gr.Image(label="Prompt frame overlay")
            propagation_gallery = gr.Gallery(label="Propagation previews", columns=3, height=220)

            video_state = gr.State()
            video_points_state = gr.State([])

            video_input.change(
                start_video_session,
                inputs=[video_input, video_state],
                outputs=[
                    frame_viewer,
                    frame_slider,
                    video_state,
                    video_points_state,
                    status_box,
                ],
            )
            refresh_frame_btn.click(
                refresh_video_frame,
                inputs=[frame_slider, video_state, video_points_state],
                outputs=[frame_viewer, status_box],
            )
            frame_viewer.select(
                handle_video_click,
                inputs=[frame_viewer, point_label_video, frame_slider, video_points_state, video_state],
                outputs=[frame_viewer, video_points_state, status_box],
            )
            clear_video_points_btn.click(
                clear_video_points,
                inputs=[frame_slider, video_state],
                outputs=[frame_viewer, video_points_state, status_box],
            )
            run_video_btn.click(
                run_video_inference,
                inputs=[
                    frame_slider,
                    text_prompt_video,
                    run_propagation_chk,
                    max_frames_slider,
                    stride_slider,
                    video_state,
                    video_points_state,
                ],
                outputs=[video_overlay, propagation_gallery, status_box],
            )

if __name__ == "__main__":
    demo.launch()
