"""Fix: add tokenizer loading back after model creation"""
with open('te_framework/llm_encoder.py', encoding='utf-8') as f:
    text = f.read()

old = """            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

        # LoRA: only Q and V projections"""

new = """            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # LoRA: only Q and V projections"""

assert old in text, "Could not find"
text = text.replace(old, new)

with open('te_framework/llm_encoder.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("Fixed")
