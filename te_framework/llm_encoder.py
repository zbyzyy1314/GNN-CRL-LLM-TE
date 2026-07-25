"""
LLM Encoder for TE — frozen LLM + LoRA adapters.

Bridges GNN structural encodings with LM semantic reasoning:
  GNN node_emb (B, N, D) → projection → soft tokens → LLM (frozen+LoRA) → reasoned tokens → project back

Usage:
    encoder = LLMEncoder(model_name="Qwen/Qwen2.5-1.5B-Instruct")
    enhanced = encoder(tm_batch, node_emb_batch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LLMEncoder(nn.Module):
    """Frozen LLM with LoRA adapters for TE reasoning enhancement.

    Args:
        hidden_dim: GNN hidden dimension (input/output dim)
        llm_dim: LLM hidden dimension (model.config.hidden_size)
        model_name: HuggingFace model name
        device: torch device
    """

    def __init__(self, hidden_dim=256, llm_dim=1536, model_name=None,
                 llm_batch_size=16, use_4bit=True, device='cuda'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.llm_dim = llm_dim
        self.device = device
        self.model_name = model_name or "Qwen/Qwen2.5-1.5B-Instruct"
        self.llm_batch_size = llm_batch_size
        self.use_4bit = use_4bit
        self._model = None
        self._tokenizer = None

        # GNN → LLM projection
        self.gnn_to_llm = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

        # LLM → GNN projection
        self.llm_to_gnn = nn.Sequential(
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Cross-attention: fuse GNN features with LLM reasoned features
        self.cross_fusion = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def load_model(self):
        """Lazy-load the LLM with 4-bit quantization and LoRA.

        Call once after init (not in __init__ to avoid import at module level).
        """
        import torch
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if self.use_4bit:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            model = prepare_model_for_kbit_training(model)
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

        # LoRA: only Q and V projections
        lora = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

        # Freeze everything except LoRA
        for name, param in model.named_parameters():
            if "lora" not in name:
                param.requires_grad = False

        model.train()  # keep train mode for LoRA
        # Disable gradient checkpointing (causes hang with 4-bit + inputs_embeds)
        model.gradient_checkpointing_disable()
        self._model = model
        self._tokenizer = tokenizer

        # Move projection layers to the same device as the LLM
        llm_device = model.device
        llm_dtype = model.dtype
        self.gnn_to_llm.to(llm_device, dtype=llm_dtype)
        self.llm_to_gnn.to(llm_device, dtype=llm_dtype)
        self.cross_fusion.to(llm_device)

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError(
                "LLM not loaded. Call encoder.load_model() before training.")
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            raise RuntimeError(
                "LLM not loaded. Call encoder.load_model() before training.")
        return self._tokenizer

    def encode(self, tm, node_emb, topo_info=None):
        """Forward pass with chunked LLM processing to fit GPU memory."""
        B, N, D = node_emb.shape
        bs = self.llm_batch_size
        all_enhanced = []
        llm_tokens = self.gnn_to_llm(node_emb)
        prompt = self._build_prompt(tm, topo_info)
        tokens = self.tokenizer(prompt, return_tensors="pt", padding=True,
                                truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            text_emb = self.model.get_input_embeddings()(tokens.input_ids)
        for start in range(0, B, bs):
            end = min(start + bs, B)
            n_chunk = llm_tokens[start:end]
            t_chunk = text_emb[start:end]
            n_emb_chunk = node_emb[start:end]
            combined = torch.cat([t_chunk, n_chunk], dim=1)
            attn_mask = torch.cat([
                tokens.attention_mask[start:end],
                torch.ones(end-start, N, dtype=torch.long, device=self.device)
            ], dim=1)
            outputs = self.model(
                inputs_embeds=combined, attention_mask=attn_mask,
                output_hidden_states=True)
            llm_out = outputs.hidden_states[-1][:, -N:, :]
            reasoned = self.llm_to_gnn(llm_out)
            enhanced, _ = self.cross_fusion(reasoned, n_emb_chunk, n_emb_chunk)
            enhanced = n_emb_chunk + enhanced
            all_enhanced.append(enhanced)
        return torch.cat(all_enhanced, dim=0)

    def _build_prompt(self, tm, topo_info=None):
        """Build WAN TE prompt for the LLM.

        Args:
            tm: (B, N, N) float tensor
            topo_info: optional dict from topology

        Returns:
            list of B prompt strings
        """
        B, N = tm.shape[:2]
        prompts = []
        for b in range(B):
            t = tm[b]
            total = t.sum().item()
            max_flow = t.max().item()
            prompt = (
                f"You are a WAN traffic engineer. "
                f"This network has {N} nodes. "
                f"Total traffic: {total:.1f}, max flow: {max_flow:.1f}. "
                f"Generate enhanced routing-aware node embeddings."
            )
            prompts.append(prompt)
        return prompts

    def forward(self, tm, node_emb, topo_info=None):
        """Alias for encode()."""
        return self.encode(tm, node_emb, topo_info)
