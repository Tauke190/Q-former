"""
BLIP-2 Stage 2: Vision-Language Generative Pre-training with OPT LLM

This is the MAIN MODEL for Stage 2 training.
Inherits from Blip2Base (vision encoder + Q-Former) and adds:
- OPT LLM decoder (frozen)
- Projector to align Q-Former output to LLM embedding space
- Text generation capabilities
"""

import torch
import torch.nn as nn
from transformers import OPTForCausalLM, AutoTokenizer

from .blip2 import Blip2Base


class Blip2OPT(Blip2Base):
    """
    BLIP-2 Stage 2 Main Model

    Inherits: Vision Encoder + Q-Former from Blip2Base
    Adds: Projector + OPT LLM

    Trainable: Q-Former + query_tokens + Projector
    Frozen: Vision Encoder + OPT LLM
    """

    def __init__(
        self,
        clip_model_name="ViT-L/14",
        opt_model_name="facebook/opt-2.7b",
        num_query_tokens=32,
        qformer_hidden_size=768,
        qformer_num_layers=12,
        qformer_num_heads=12,
        cross_attention_freq=2,
        max_txt_len=128,
        prompt="",
        device="cuda",
    ):
        # Initialize Blip2Base (vision encoder + Q-Former only)
        super().__init__(
            clip_model_name=clip_model_name,
            num_query_tokens=num_query_tokens,
            qformer_hidden_size=qformer_hidden_size,
            qformer_num_layers=qformer_num_layers,
            qformer_num_heads=qformer_num_heads,
            cross_attention_freq=cross_attention_freq,
            device=device,
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt

        # ========== OPT LLM (Frozen) ==========
        self.opt_model = OPTForCausalLM.from_pretrained(opt_model_name)
        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model_name, use_fast=False)

        for param in self.opt_model.parameters():
            param.requires_grad = False

        self.llm_hidden_size = self.opt_model.config.hidden_size

        # ========== Projector ==========
        # Maps Q-Former output (768) to OPT embedding space (2560 for opt-2.7b)
        self.llm_proj = nn.Linear(qformer_hidden_size, self.llm_hidden_size)

    def forward(self, image, text):
        """
        Forward pass for Stage 2 training.
        Computes only Language Modeling (LM) loss.

        Args:
            image: PIL Image or tensor [B, C, H, W]
            text: List of strings or dict with input_ids/attention_mask

        Returns:
            dict with loss_lm
        """
            # Ensure image tensor dtype matches model weights
        if isinstance(image, torch.Tensor):
            model_dtype = next(self.parameters()).dtype
            if image.dtype != model_dtype:
                image = image.to(dtype=model_dtype)
        image_embeds, image_atts = self.encode_image(image)

        # Tokenize text for OPT
        if isinstance(text, (list, str)):
            text_tokens = self.opt_tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.device)
            text_input_ids = text_tokens.input_ids
            text_attention_mask = text_tokens.attention_mask
        else:
            text_input_ids = text["input_ids"].to(self.device)
            text_attention_mask = text["attention_mask"].to(self.device)

        loss_lm = self.compute_lm_loss(image_embeds, image_atts, text_input_ids, text_attention_mask)

        return {
            "loss": loss_lm,
            "loss_lm": loss_lm,
        }

    def compute_lm_loss(self, image_embeds, image_atts, text_input_ids, text_attention_mask):
        """
        Language Modeling loss for Stage 2.

        Args:
            image_embeds: Visual features [B, N, D]
            image_atts: Image attention mask [B, N]
            text_input_ids: Text token ids [B, L]
            text_attention_mask: Text attention mask [B, L]

        Returns:
            LM loss
        """
        batch_size = image_embeds.size(0)
        device = image_embeds.device

        # Get Q-Former output and project to LLM space
        query_output = self.get_query_features(image_embeds, image_atts)
        inputs_llm = self.llm_proj(query_output)

        # Query attention mask
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long, device=device)

        # Get text embeddings from OPT
        inputs_embeds = self.opt_model.model.decoder.embed_tokens(text_input_ids)

        # Ensure dtype consistency between projected image features and text embeddings
        inputs_llm = inputs_llm.to(dtype=inputs_embeds.dtype)

        # Concatenate: [query_embeds, text_embeds]
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        combined_attention_mask = torch.cat([atts_llm, text_attention_mask], dim=1)

        # Prepare labels (pad query positions with -100)
        labels = text_input_ids.clone()
        empty_labels = torch.full(
            (batch_size, self.num_query_tokens),
            fill_value=-100,
            dtype=torch.long,
            device=device,
        )
        labels = torch.cat([empty_labels, labels], dim=1)

        # Forward through OPT
        outputs = self.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            labels=labels,
            return_dict=True,
        )

        return outputs.loss

    @torch.no_grad()
    def generate(
        self,
        image,
        prompt="",
        max_new_tokens=50,
        num_beams=5,
        top_p=0.9,
        temperature=1.0,
        repetition_penalty=1.0,
    ):
        """
        Generate text conditioned on image.

        Args:
            image: PIL Image or tensor
            prompt: Optional text prompt
            max_new_tokens: Maximum tokens to generate

        Returns:
            List of generated strings
        """
        image_embeds, image_atts = self.encode_image(image)
        batch_size = image_embeds.size(0)
        device = image_embeds.device

        # Get projected query features
        query_output = self.get_query_features(image_embeds, image_atts)
        inputs_llm = self.llm_proj(query_output)
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long, device=device)

        # Process prompt
        if prompt:
            prompt = [prompt] * batch_size if isinstance(prompt, str) else prompt
            prompt_tokens = self.opt_tokenizer(
                prompt, return_tensors="pt", padding=True
            ).to(device)
            prompt_embeds = self.opt_model.model.decoder.embed_tokens(
                prompt_tokens.input_ids
            )
            # Ensure dtype consistency
            inputs_llm = inputs_llm.to(dtype=prompt_embeds.dtype)
            inputs_embeds = torch.cat([inputs_llm, prompt_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, prompt_tokens.attention_mask], dim=1)
        else:
            # For generation without prompt, match OPT embedding dtype
            opt_dtype = self.opt_model.model.decoder.embed_tokens.weight.dtype
            inputs_llm = inputs_llm.to(dtype=opt_dtype)
            inputs_embeds = inputs_llm
            attention_mask = atts_llm

        # Generate
        outputs = self.opt_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            do_sample=top_p < 1.0,
        )

        return self.opt_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def get_trainable_parameters(self):
        """
        Returns only trainable parameters for optimizer.

        Trainable: Q-Former + query_tokens + Projector
        Frozen: Vision Encoder + OPT LLM
        """
        trainable = []

        # Q-Former parameters
        for param in self.Qformer.parameters():
            if param.requires_grad:
                trainable.append(param)

        # Query tokens
        trainable.append(self.query_tokens)

        # Projector
        trainable.extend(self.llm_proj.parameters())

        return trainable
