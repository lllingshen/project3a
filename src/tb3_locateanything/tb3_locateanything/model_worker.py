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

if __package__:
    from .inventory_parser import (model_categories, normalize_categories,
                                   parse_inventory_output, parse_inventory_response)
else:
    from inventory_parser import (model_categories, normalize_categories,
                                  parse_inventory_output, parse_inventory_response)


def emit(payload):
    print(json.dumps(payload), flush=True)


def model_prompt(operation, expression="", categories=None):
    """Use Eagle's single-instance grounding or class-detection prompt."""
    if operation == "inventory":
        labels = "</c>".join(model_categories(categories))
        return f"Locate all the instances that matches the following description: {labels}."
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression must be non-empty text")
    if operation == "ground":
        return f"Locate a single instance that matches the following description: {expression}."
    if operation == "catalog":
        return f"Locate all the instances that matches the following description: {expression}."
    raise ValueError(f"unsupported model operation: {operation}")


def infer_request(request, width, height, generate):
    """Run a joint prompt, then at most one retry per missing category.

    ``generate(prompt, multi_instance)`` uses the same decoded image for every
    call and returns its exact answer and measured generation time. Keeping this
    protocol separate lets tests verify the bounded retry without loading CUDA.
    Only configured categories without primary boxes are retried, using their
    canonical names. Total calls are bounded by one plus the category count;
    supplements do not trigger another round or certify an object's identity.
    """
    operation = request.get("operation", "ground")
    phrase = request.get("expression", "")
    categories = request.get("categories")
    prompt = model_prompt(operation, phrase, categories)
    multi_instance = operation in ("catalog", "inventory")
    answer, primary_seconds = generate(prompt, multi_instance)
    answer = str(answer)
    supplements = []
    if operation == "inventory":
        categories = normalize_categories(categories)
        parsed = parse_inventory_output(answer, width, height, categories)
        present = {item["label"] for item in parsed["observations"]}
        for category in categories:
            if category.replace(" ", "_") in present:
                continue
            supplement_prompt = model_prompt("catalog", category)
            extra_answer, extra_seconds = generate(supplement_prompt, True)
            supplements.append({"category": category, "prompt": supplement_prompt,
                                "answer": str(extra_answer), "inference_seconds": extra_seconds})
    total_seconds = primary_seconds + sum(item["inference_seconds"] for item in supplements)
    response = {"status": "ok", "request_id": request["request_id"],
                "operation": operation, "command": request.get("command", ""),
                "expression": phrase, "prompt": prompt,
                "categories": categories if operation == "inventory" else None,
                "generation_mode": "slow" if multi_instance else "hybrid",
                "answer": answer, "supplemental_inventory": supplements,
                "primary_inference_seconds": primary_seconds,
                "total_inference_seconds": total_seconds, "inference_seconds": total_seconds,
                "image_width": width, "image_height": height}
    if operation == "inventory":
        parse_inventory_response(response, width, height, categories)
    return response


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

            def generate(prompt, multi_instance):
                started = time.monotonic()
                with contextlib.redirect_stdout(sys.stderr):
                    torch.manual_seed(0)
                    torch.cuda.manual_seed_all(0)
                    # Deterministic call from run_locateanything_base.py;
                    # Eagle.predict samples. Both calls use this exact image.
                    messages = [{"role": "user", "content": [
                        {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
                    text = worker.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    images, videos = worker.processor.process_vision_info(messages)
                    inputs = worker.processor(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
                    with torch.no_grad():
                        response = worker.model.generate(
                            pixel_values=inputs["pixel_values"].to(torch.bfloat16), input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"], image_grid_hws=inputs.get("image_grid_hws"),
                            tokenizer=worker.tokenizer,
                            max_new_tokens=512 if multi_instance else 256, use_cache=True,
                            # Hybrid repeats catalog boxes; slow emits EOS.
                            generation_mode="slow" if multi_instance else "hybrid",
                            do_sample=False, repetition_penalty=1.1, verbose=False)
                    torch.cuda.synchronize()
                    answer = response[0] if isinstance(response, tuple) else response
                return str(answer), time.monotonic() - started

            emit(infer_request(request, image.width, image.height, generate))
        except Exception as exc:
            emit({"status": "error", "request_id": request.get("request_id"),
                  "operation": request.get("operation", "ground"), "error": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
