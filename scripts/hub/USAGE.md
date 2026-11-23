# Run a decision locally

Download the **whole model repository**, not just `model.safetensors`. The weights contain the numbers the model learned. The included `run_joey.py` tells it which layers to use twice. The same command works for both models, including Compact Loop without access to the larger checkpoint.

**Check the download on any CPU** with `python run_joey.py --verify-only` from the snapshot directory (Python 3.10+). It checks the pinned weight file, tokenizer, configuration, layer order and OneJev sources. It does not run the model.

**For inference**, use a Linux CUDA environment matching the [upstream Joey setup](https://huggingface.co/baselabs/joey-1/blob/db5c2a631f2da93c4173378438d8246966e25159/setup.sh). The reference is `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` with its matching torchvision, one NVIDIA GPU and BF16 support. In that environment, from the downloaded directory:

```bash
# Keep the image's matched CUDA builds when installing the inference dependencies.
python -c 'import torch, torchvision; print(f"torch=={torch.__version__}\ntorchvision=={torchvision.__version__}")' > image-torch.txt
python -m pip install -c constraints.txt -c image-torch.txt \
  transformers accelerate tokenizers safetensors pydantic 'jinja2>=3.1' \
  huggingface_hub flash-linear-attention ninja
python -m pip install -c constraints.txt -c image-torch.txt --no-build-isolation causal-conv1d
python run_joey.py
```

The last command scores a sample support ticket. To supply your own, save a `request.json` like this:

```json
{
  "state": "My bank card expires next week. How can I replace it?",
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this request?",
      "criteria": {"cards": "Issuing or replacing a card", "transfers": "Sending money"}
    }
  }
}
```

Run `python run_joey.py --request request.json`. Look under `team` for `choice` and `probabilities`. Scores are for the supplied choices only. The runner uses local files, checks the native one-pass readout before enabling the loop, and makes no network request for inference. The first load includes verification; do not use process startup as a per-decision latency measurement. No image input, open-ended text generation, confidence threshold, or hosted Inference API is promised by this release.
