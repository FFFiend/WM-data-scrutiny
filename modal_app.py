"""
Modal GPU inference services for all models.

Three classes, each loads its model once per container:
  - QwenVLM:       Qwen2.5-VL structured tagging
  - GroundingDINO: open-vocab object detection
  - DINOv2Embed:   frame embedding for domain gap

Deploy:   modal deploy modal_app.py
Test:     modal run modal_app.py
"""

import io
import json
import modal

app = modal.App("wm-data-scrutiny")

# ── Shared volume for HF model weights ──────────────────────────────
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"

# ── Images ──────────────────────────────────────────────────────────
vlm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "transformers>=4.40", "accelerate",
        "qwen-vl-utils", "Pillow", "numpy",
    )
)

dino_det_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "transformers>=4.40",
        "Pillow", "numpy",
    )
)

dino_embed_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "Pillow", "numpy", "timm",
    )
)


# ═══════════════════════════════════════════════════════════════════
# 1. Qwen2.5-VL  —  structured taxonomy tagging
# ═══════════════════════════════════════════════════════════════════

@app.cls(
    image=vlm_image,
    gpu="A100",
    timeout=600,
    volumes={HF_CACHE_PATH: hf_cache},
    scaledown_window=120,
)
class QwenVLM:
    @modal.enter()
    def load(self):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto",
        )
        self.model.eval()

    @modal.method()
    def tag_frames(self, frame_bytes_list: list[bytes], prompt: str) -> str:
        """
        Args:
            frame_bytes_list: list of PNG-encoded frame bytes.
            prompt: the structured tagging prompt.
        Returns:
            Raw model response text (parsed locally).
        """
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        image_content = []
        for fb in frame_bytes_list:
            img = Image.open(io.BytesIO(fb)).convert("RGB")
            image_content.append({"type": "image", "image": img})
        image_content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": image_content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=512, temperature=0.1, do_sample=False,
            )
        generated = output_ids[0][inputs.input_ids.shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()


# ═══════════════════════════════════════════════════════════════════
# 2. Grounding DINO  —  open-vocab object detection
# ═══════════════════════════════════════════════════════════════════

@app.cls(
    image=dino_det_image,
    gpu="A10G",
    timeout=300,
    volumes={HF_CACHE_PATH: hf_cache},
    scaledown_window=120,
)
class GroundingDINODetector:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        model_id = "IDEA-Research/grounding-dino-base"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    @modal.method()
    def detect(
        self,
        frame_bytes_list: list[bytes],
        query_text: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> list[dict]:
        """
        Returns per-frame detection results:
        [{"labels": [...], "scores": [...], "n_objects": int}, ...]
        """
        import torch
        from PIL import Image

        results = []
        for fb in frame_bytes_list:
            img = Image.open(io.BytesIO(fb)).convert("RGB")
            inputs = self.processor(
                images=img, text=query_text, return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            try:
                post = self.processor.post_process_grounded_object_detection(
                    outputs, inputs["input_ids"],
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=[img.size[::-1]],
                )[0]
            except TypeError:
                # Newer transformers API uses 'threshold' instead
                post = self.processor.post_process_grounded_object_detection(
                    outputs, inputs["input_ids"],
                    threshold=box_threshold,
                    target_sizes=[img.size[::-1]],
                )[0]

            labels = post.get("labels", post.get("text", []))
            scores = post.get("scores", [])
            if hasattr(scores, "tolist"):
                scores = scores.tolist()

            results.append({
                "labels": [l.lower().strip() for l in labels],
                "scores": scores,
                "n_objects": len(labels),
            })

        return results


# ═══════════════════════════════════════════════════════════════════
# 3. DINOv2  —  frame embeddings for domain gap analysis
# ═══════════════════════════════════════════════════════════════════

@app.cls(
    image=dino_embed_image,
    gpu="A10G",
    timeout=300,
    volumes={HF_CACHE_PATH: hf_cache},
    scaledown_window=120,
)
class DINOv2Embed:
    @modal.enter()
    def load(self):
        import torch

        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.model.eval()
        self.model = self.model.cuda()

        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @modal.method()
    def embed_frames(self, frame_bytes_list: list[bytes]) -> list[list[float]]:
        """
        Embed each frame independently.
        Returns list of embedding vectors (each 768-d).
        """
        import torch
        from PIL import Image

        embeddings = []
        for fb in frame_bytes_list:
            img = Image.open(io.BytesIO(fb)).convert("RGB")
            tensor = self.transform(img).unsqueeze(0).cuda()
            with torch.no_grad():
                emb = self.model(tensor)
            embeddings.append(emb.squeeze().cpu().tolist())

        return embeddings


# ── Test entrypoint ─────────────────────────────────────────────────

@app.local_entrypoint()
def test():
    """Quick smoke test: embed a blank image."""
    from PIL import Image

    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    embedder = DINOv2Embed()
    result = embedder.embed_frames.remote([img_bytes])
    print(f"DINOv2 embedding dim: {len(result[0])}")
    print("Modal smoke test passed.")
