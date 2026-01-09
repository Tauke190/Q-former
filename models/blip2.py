"""
BLIP-2 Base Model

Initializes the vision encoder (CLIP) and Q-Former.
This is the base class inherited by Stage 1 and Stage 2 models.
"""

import torch
import torch.nn as nn
import clip
from transformers import BertTokenizer, BertConfig

from .Qformer import BertModel, BertLMHeadModel


class Blip2Base(nn.Module):
    """
    Base BLIP-2 Model: Vision Encoder + Q-Former

    This class is inherited by:
    - Blip2Qformer (Stage 1 training)
    - Blip2OPT (Stage 2 training with LLM)
    """

    def __init__(
        self,
        clip_model_name="ViT-L/14",
        num_query_tokens=32,
        qformer_hidden_size=768,
        qformer_num_layers=12,
        qformer_num_heads=12,
        cross_attention_freq=2,
        device="cuda",
    ):
        super().__init__()

        self.device = device
        self.num_query_tokens = num_query_tokens
        self.qformer_hidden_size = qformer_hidden_size

        # ========== Vision Encoder (Frozen) ==========
        self.clip_model, self.preprocess = clip.load(clip_model_name, device=device)
        self.vision_width = self.clip_model.visual.output_dim

        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

        # ========== Q-Former ==========
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, qformer_hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=0.02)

        self.qformer_config = BertConfig(
            vocab_size=30522,
            hidden_size=qformer_hidden_size,
            num_hidden_layers=qformer_num_layers,
            num_attention_heads=qformer_num_heads,
            intermediate_size=qformer_hidden_size * 4,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            encoder_width=self.vision_width,
            add_cross_attention=True,
            cross_attention_freq=cross_attention_freq,
            query_length=num_query_tokens,
        )

        self.Qformer = BertModel(self.qformer_config, add_pooling_layer=False)

        # Q-Former with LM head (for ITG in Stage 1)
        self.Qformer_lm = BertLMHeadModel(self.qformer_config)
        self.Qformer_lm.bert = self.Qformer  # Share weights

        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        self.Qformer.resize_token_embeddings(len(self.tokenizer))

    @torch.no_grad()
    def encode_image(self, image):
        """
        Encode image using frozen CLIP vision encoder.

        Args:
            image: PIL Image or tensor [B, C, H, W]

        Returns:
            image_embeds: [B, num_patches+1, vision_width]
            image_atts: [B, num_patches+1]
        """
        if not isinstance(image, torch.Tensor):
            image = self.preprocess(image).unsqueeze(0)

        image = image.to(self.device)

        # Get intermediate features from CLIP ViT
        x = self.clip_model.visual.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([
            self.clip_model.visual.class_embedding.expand(x.shape[0], -1, -1),
            x
        ], dim=1)
        x = x + self.clip_model.visual.positional_embedding
        x = self.clip_model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.visual.transformer(x)
        x = x.permute(1, 0, 2)
        image_embeds = self.clip_model.visual.ln_post(x)

        image_atts = torch.ones(image_embeds.size()[:-1], device=self.device)
        return image_embeds, image_atts

    def get_query_features(self, image_embeds, image_atts):
        """
        Get Q-Former query output features.

        Args:
            image_embeds: Visual features [B, N, D]
            image_atts: Image attention mask [B, N]

        Returns:
            Query output features [B, num_query_tokens, qformer_hidden_size]
        """
        batch_size = image_embeds.size(0)
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        query_output = self.Qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state
