"""
BLIP-2 Stage 2: Vision-Language Generative Pre-training with OPT LLM

This is the MAIN MODEL for Stage 2 training.
Inherits from Blip2Qformer (vision encoder + Q-Former + ITC/ITM/ITG) and adds:
- OPT LLM decoder (frozen)
- Projector to align Q-Former output to LLM embedding space
"""

import torch
import torch.nn as nn
from transformers import OPTForCausalLM, AutoTokenizer

from .blip2_qformer import Blip2Qformer


class Blip2OPT(Blip2Qformer):
    """
    BLIP-2 Stage 2 Main Model

    Inherits: Vision Encoder + Q-Former + ITC/ITM/ITG from Blip2Qformer
    Adds: Projector + OPT LLM

    Trainable: Q-Former + Projector
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
        # Initialize Blip2Qformer (vision encoder + Q-Former + Stage 1 heads)
        super().__init__(
            clip_model_name=clip_model_name,
            num_query_tokens=num_query_tokens,
            qformer_hidden_size=qformer_hidden_size,
            qformer_num_layers=qformer_num_layers,
            qformer_num_heads=qformer_num_heads,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
            device=device,
        )

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

    def forward_lm(self, image_embeds, image_atts, text_input_ids, text_attention_mask):
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

    def forward(self, image, text):
        """
        Forward pass for Stage 2 training.
        Combines ITC + ITM + ITG (from Blip2Qformer) + LM loss.

        Args:
            image: PIL Image or tensor [B, C, H, W]
            text: List of strings or dict with input_ids/attention_mask

        Returns:
            dict with all losses
        """
        # Encode image
        image_embeds, image_atts = self.encode_image(image)

        # Tokenize text for Q-Former (BERT tokenizer)
        if isinstance(text, (list, str)):
            text_tokens = self.tokenizer(
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

        # Stage 1 losses (ITC, ITM, ITG)
        loss_itc, image_feats, text_feats = self.forward_itc(
            image_embeds, image_atts, text_input_ids, text_attention_mask
        )

        loss_itm = self.forward_itm(
            image_embeds, image_atts, text_input_ids, text_attention_mask,
            image_feats, text_feats
        )

        loss_itg = self.forward_itg(
            image_embeds, image_atts, text_input_ids, text_attention_mask
        )

        # Stage 2: Tokenize for OPT and compute LM loss
        if isinstance(text, (list, str)):
            opt_tokens = self.opt_tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.device)
            opt_input_ids = opt_tokens.input_ids
            opt_attention_mask = opt_tokens.attention_mask
        else:
            # Assume pre-tokenized for OPT
            opt_input_ids = text["opt_input_ids"].to(self.device)
            opt_attention_mask = text["opt_attention_mask"].to(self.device)

        loss_lm = self.forward_lm(
            image_embeds, image_atts, opt_input_ids, opt_attention_mask
        )

        total_loss = loss_itc + loss_itm + loss_itg + loss_lm

        return {
            "loss": total_loss,
            "loss_itc": loss_itc,
            "loss_itm": loss_itm,
            "loss_itg": loss_itg,
            "loss_lm": loss_lm,
        }

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
            inputs_embeds = torch.cat([inputs_llm, prompt_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, prompt_tokens.attention_mask], dim=1)
        else:
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

        Trainable: Q-Former + query_tokens + Stage1 heads + Projector
        """
        trainable = []

        # Q-Former parameters
        for param in self.Qformer.parameters():
            if param.requires_grad:
                trainable.append(param)

        # Query tokens
        trainable.append(self.query_tokens)

        # Stage 1 heads
        trainable.extend(self.vision_proj.parameters())
        trainable.extend(self.text_proj.parameters())
        trainable.extend(self.itm_head.parameters())
        trainable.append(self.temp)

        # Projector
        trainable.extend(self.llm_proj.parameters())

        return trainable
