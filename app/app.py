"""Gradio demo: upload a weld radiograph tile -> predicted class, defect score, Grad-CAM.

Run from the project folder:
    python app/app.py                       # uses checkpoints/resnet50_clean_seed42/best.pth
    python app/app.py --checkpoint path/to/best.pth --share
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from weld_insp.config import CLASSES, CLASS_NAMES  # noqa: E402
from weld_insp.data import build_transforms  # noqa: E402
from weld_insp.train import build_model, device  # noqa: E402


def load(checkpoint: Path, arch: str = "resnet50"):
    import torch
    model = build_model(arch, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.to(device()).eval()


def make_predict_fn(model):
    import torch
    from PIL import Image
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    tf = build_transforms(train=False)
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

    def predict(img):
        if img is None:
            return {}, "", None
        pil = Image.fromarray(np.asarray(img)).convert("L")
        x = tf(pil).unsqueeze(0).to(device())
        with torch.no_grad():
            prob = torch.softmax(model(x), 1)[0].cpu().numpy()
        heat = cam(input_tensor=x)[0]
        base = np.asarray(pil.resize((224, 224))).astype(np.float32) / 255
        overlay = show_cam_on_image(np.stack([base] * 3, -1), heat, use_rgb=True)
        labels = {f"{c} ({CLASS_NAMES[c]})": float(p) for c, p in zip(CLASSES, prob)}
        score = 1 - float(prob[CLASSES.index("ND")])
        return labels, f"Defect score P(defect) = {score:.3f}", overlay

    return predict


def build_demo(checkpoint: Path):
    import gradio as gr
    predict = make_predict_fn(load(checkpoint))
    examples_dir = ROOT / "app" / "examples"
    examples = [[str(p)] for p in sorted(examples_dir.glob("*.png"))] if examples_dir.exists() else None
    return gr.Interface(
        fn=predict,
        inputs=gr.Image(label="Weld radiograph tile (grayscale PNG)"),
        outputs=[gr.Label(num_top_classes=4, label="Predicted class"), gr.Textbox(label="Defect score"),
                 gr.Image(label="Grad-CAM: where the model looked")],
        title="weld_insp: weld defect classifier",
        description="ResNet50 fine-tuned on RIAWELC (leakage-free split). Classes: crack, porosity, "
                    "lack of penetration, no defect. Research demo, not a certified inspection tool.",
        examples=examples,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "resnet50_clean_seed42" / "best.pth"))
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    build_demo(Path(a.checkpoint)).launch(share=a.share)
