"""The specialist: our own QLoRA fine-tune, loaded locally.

Needs a CUDA GPU and an adapter produced by `notebooks/4_qlora_finetune_colab.ipynb`. On a CPU-only box
this raises immediately rather than pretending -- use `ClassicalAgent` there instead.
"""

import os
import re

from pricer.agents.agent import Agent
from pricer.items import PREFIX
from pricer.prompts import render

BASE_MODEL = "Qwen/Qwen2.5-3B"
ADAPTER = os.environ.get("WINE_ADAPTER", "borjahernandez/wine-pricer-qlora")
NUMBER = re.compile(r"[-+]?\d*\.\d+|\d+")


class SpecialistAgent(Agent):
    name = "Specialist Agent"
    colour = "\033[32m"

    def __init__(self, base: str = BASE_MODEL, adapter: str = ADAPTER):
        import torch  # imported here so the CPU-only agents do not pay for it
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("SpecialistAgent needs a CUDA GPU; run it in Colab or use ClassicalAgent")
        self.log(f"Loading {base} in 4-bit with adapter {adapter}")
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        weights = AutoModelForCausalLM.from_pretrained(base, quantization_config=quantization, device_map="auto")
        self.model = PeftModel.from_pretrained(weights, adapter)
        self.model.eval()
        self.log("Ready")

    def price(self, text: str) -> float:
        prompt = render(text)  # the same layout the fine-tune was trained on, minus the answer
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=6, do_sample=False)
        reply = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split(PREFIX)[-1]
        match = NUMBER.search(reply)
        guess = float(match.group()) if match else 0.0
        self.log(f"Estimated ${guess:.2f}")
        return guess
