"""
BLIP-2 Stage 1: Q-Former Pre-training Wrapper

Adds training objectives on top of the base model:
- Image-Text Contrastive Learning (ITC)
- Image-Text Matching (ITM)
- Image-grounded Text Generation (ITG)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blip2 import Blip2Base


class Blip2Qformer(Blip2Base):
    """
    Stage 1 Training Model: Inherits from Blip2Base

    Adds ITC, ITM, ITG training objectives for Q-Former pre-training.
    """

    def __init__(
        self,
        clip_model_name="ViT-L/14",
        num_query_tokens=32,
        qformer_hidden_size=768,
        qformer_num_layers=12,
        qformer_num_heads=12,
        cross_attention_freq=2,
        max_txt_len=32,
        device="cuda",
    ):
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

        # ========== Stage 1 Specific Heads ==========
        # Projection heads for contrastive learning
        self.vision_proj = nn.Linear(qformer_hidden_size, 256)
        self.text_proj = nn.Linear(qformer_hidden_size, 256)

        # ITM classification head
        self.itm_head = nn.Linear(qformer_hidden_size, 2)

        # Learnable temperature for contrastive loss
        self.temp = nn.Parameter(0.07 * torch.ones([]))

    def forward_itc(self, image_embeds, image_atts, text_input_ids, text_attention_mask):
        """Image-Text Contrastive Learning"""
        batch_size = image_embeds.size(0)
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        # Image features through Q-Former
        query_output = self.Qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state[:, 0, :]), dim=-1
        )

        # Text features through Q-Former (no cross-attention)
        text_output = self.Qformer(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            return_dict=True,
        )
        text_feats = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        # InfoNCE contrastive loss
        sim_i2t = image_feats @ text_feats.t() / self.temp
        sim_t2i = text_feats @ image_feats.t() / self.temp

        targets = torch.arange(batch_size, device=image_embeds.device)
        loss_itc = (
            F.cross_entropy(sim_i2t, targets) + F.cross_entropy(sim_t2i, targets)
        ) / 2

        return loss_itc, image_feats, text_feats

    def forward_itm(self, image_embeds, image_atts, text_input_ids, text_attention_mask,
                    image_feats, text_feats):
        """Image-Text Matching with hard negative mining"""
        batch_size = image_embeds.size(0)
        device = image_embeds.device

        # Hard negative mining
        with torch.no_grad():
            sim_i2t = image_feats @ text_feats.t()
            sim_t2i = text_feats @ image_feats.t()

            mask = torch.eye(batch_size, device=device).bool()
            sim_i2t.masked_fill_(mask, -1e9)
            sim_t2i.masked_fill_(mask, -1e9)

            weights_i2t = F.softmax(sim_i2t, dim=1)
            weights_t2i = F.softmax(sim_t2i, dim=1)

            neg_text_idx = torch.multinomial(weights_i2t, 1).squeeze()
            neg_image_idx = torch.multinomial(weights_t2i, 1).squeeze()

        query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        # Positive pairs
        pos_output = self.Qformer(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        pos_logits = self.itm_head(pos_output.last_hidden_state[:, 0, :])

        # Negative: image with wrong text
        neg_output_t = self.Qformer(
            input_ids=text_input_ids[neg_text_idx],
            attention_mask=text_attention_mask[neg_text_idx],
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        neg_logits_t = self.itm_head(neg_output_t.last_hidden_state[:, 0, :])

        # Negative: text with wrong image
        neg_output_i = self.Qformer(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds[neg_image_idx],
            encoder_attention_mask=image_atts[neg_image_idx],
            return_dict=True,
        )
        neg_logits_i = self.itm_head(neg_output_i.last_hidden_state[:, 0, :])

        logits = torch.cat([pos_logits, neg_logits_t, neg_logits_i], dim=0)
        labels = torch.cat([
            torch.ones(batch_size, device=device),
            torch.zeros(batch_size * 2, device=device),
        ]).long()

        return F.cross_entropy(logits, labels)

    def forward_itg(self, image_embeds, image_atts, text_input_ids, text_attention_mask):
        """Image-grounded Text Generation"""
        batch_size = image_embeds.size(0)

        decoder_input_ids = text_input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id

        labels = text_input_ids.masked_fill(
            text_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], device=image_embeds.device)
        attention_mask = torch.cat([query_atts, text_attention_mask], dim=1)

        output = self.Qformer_lm(
            input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            labels=labels,
            return_dict=True,
            is_decoder=True,
        )

        return output.loss

    def forward(self, image, text):
        """
        Forward pass for Stage 1 training.

        Args:
            image: PIL Image or tensor [B, C, H, W]
            text: List of strings or dict with input_ids/attention_mask

        Returns:
            dict with loss, loss_itc, loss_itm, loss_itg
        """
        # Encode image using base class method
        image_embeds, image_atts = self.encode_image(image)

        # Tokenize text
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

        # Three training objectives
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

        total_loss = loss_itc + loss_itm + loss_itg

        return {
            "loss": total_loss,
            "loss_itc": loss_itc,
            "loss_itm": loss_itm,
            "loss_itg": loss_itg,
        }
