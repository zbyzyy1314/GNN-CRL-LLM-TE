"""Disable gradient checkpointing in LLM encoder to fix 4-bit hang."""
with open('te_framework/llm_encoder.py', encoding='utf-8') as f:
    text = f.read()

# After prepare_model_for_kbit_training, disable gradient_checkpointing
old = ("        )\n"
       "        model.train()  # keep train mode for LoRA")

new = ("        )\n"
       "        model.config.use_cache = True\n"
       "        # Disable gradient checkpointing (incompatible with 4-bit + inputs_embeds)\n"
       "        if hasattr(model, 'gradient_checkpointing_enable'):\n"
       "            try: model.gradient_checkpointing_disable()\n"
       "            except: pass\n"
       "        model.train()  # keep train mode for LoRA")

assert old in text, "Could not find"
text = text.replace(old, new)

with open('te_framework/llm_encoder.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("Fixed checkpointing")
