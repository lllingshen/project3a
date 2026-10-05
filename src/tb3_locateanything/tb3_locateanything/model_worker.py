#!/usr/bin/env python3
"""Persistent JSON-lines worker. Run with the isolated model Python, not ROS.

Imports the user's external Eagle installation; no model/runtime code is copied.
The command's complete referring expression reaches the model prompt. stdout is
reserved for IPC; third-party progress output is redirected to stderr.
"""
import argparse
import base64
import contextlib
import io
import json
from pathlib import Path
import sys
import time


def emit(payload):
    print(json.dumps(payload), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--eagle-path", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import torch
            from PIL import Image
            eagle_path = Path(args.eagle_path)
            if (eagle_path / "Embodied" / "locateanything_worker.py").is_file():
                eagle_path = eagle_path / "Embodied"
            sys.path.insert(0, str(eagle_path))
            from locateanything_worker import LocateAnythingWorker
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable in worker environment")
            worker = LocateAnythingWorker(args.model_path, device="cuda", dtype=torch.bfloat16)
        emit({"status": "ready", "load_seconds": time.monotonic() - started})
    except Exception as exc:
        emit({"status": "error", "error": str(exc)})
        return 1
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            if request.get("op") == "stop":
                break
            image = Image.open(io.BytesIO(base64.b64decode(request["image_png"]))).convert("RGB")
            phrase = request["expression"]
            prompt = f"Locate a single instance that matches the following description: {phrase}."
            started = time.monotonic()
            with contextlib.redirect_stdout(sys.stderr):
                torch.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                # Deterministic processor/generation call adapted from the
                # research run_locateanything_base.py; Eagle.predict samples.
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
                text = worker.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                images, videos = worker.processor.process_vision_info(messages)
                inputs = worker.processor(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    response = worker.model.generate(
                        pixel_values=inputs["pixel_values"].to(torch.bfloat16), input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"], image_grid_hws=inputs.get("image_grid_hws"),
                        tokenizer=worker.tokenizer, max_new_tokens=256, use_cache=True,
                        generation_mode="hybrid", do_sample=False, repetition_penalty=1.1, verbose=False)
                torch.cuda.synchronize()
                answer = response[0] if isinstance(response, tuple) else response
            emit({"status": "ok", "request_id": request["request_id"],
                  "command": request["command"], "expression": phrase, "prompt": prompt,
                  "answer": str(answer),
                  "inference_seconds": time.monotonic() - started,
                  "image_width": image.width, "image_height": image.height})
        except Exception as exc:
            emit({"status": "error", "request_id": request.get("request_id"), "error": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
