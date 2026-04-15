"""
BARTpho Multi-Task Learning Model.

Architecture:
  - Shared encoder:  BARTpho-syllable-base encoder
  - Task 1 head:     NSW Detection  (token classification → 2 classes)
  - Task 2:          Normalization   (BARTpho decoder + LM head)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MBartForConditionalGeneration


class BARTphoMTL(nn.Module):
    """
    Multi-Task Learning model built on BARTpho-syllable-base.

    Supports three modes:
      - "detection_only":      Only NSW detection head is active
      - "normalization_only":  Only seq2seq normalization is active
      - "mtl":                 Both tasks are active
    """

    def __init__(self, model_name: str = "vinai/bartpho-syllable-base", mode: str = "mtl"):
        super().__init__()
        self.mode = mode

        # Load pretrained BARTpho (uses mBART architecture)
        self.bartpho = MBartForConditionalGeneration.from_pretrained(model_name)

        # Detection head: encoder hidden → 2 classes (standard / NSW)
        hidden_size = self.bartpho.config.d_model  # 768 for base
        self.detection_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2),
        )

        # Initialize detection head weights
        self._init_detection_head()

    def _init_detection_head(self):
        """Xavier initialization for detection head."""
        for module in self.detection_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    @property
    def encoder(self):
        return self.bartpho.get_encoder()

    @property
    def decoder(self):
        return self.bartpho.get_decoder()

    @property
    def lm_head(self):
        return self.bartpho.lm_head

    def get_shared_params(self):
        """Return list of shared encoder parameters (for PCGrad)."""
        return list(self.encoder.parameters())

    def get_detection_params(self):
        """Return detection head parameters."""
        return list(self.detection_head.parameters())

    def get_normalization_params(self):
        """Return decoder + lm_head parameters."""
        return list(self.decoder.parameters()) + list(self.lm_head.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        detection_labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for MTL.

        Returns a dict with available keys:
          - det_logits, det_loss    (if detection is active)
          - norm_logits, norm_loss  (if normalization is active)
        """
        # ── Shared encoder ─────────────────────────────────────────
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = encoder_outputs.last_hidden_state  # (B, S, D)

        outputs = {"encoder_hidden_states": hidden_states}

        # ── Task 1: NSW Detection ──────────────────────────────────
        if self.mode in ("detection_only", "mtl"):
            det_logits = self.detection_head(hidden_states)  # (B, S, 2)
            outputs["det_logits"] = det_logits

            if detection_labels is not None:
                det_loss = F.cross_entropy(
                    det_logits.view(-1, 2),
                    detection_labels.view(-1),
                    ignore_index=-100,
                )
                outputs["det_loss"] = det_loss

        # ── Task 2: Normalization ──────────────────────────────────
        if self.mode in ("normalization_only", "mtl"):
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                return_dict=True,
            )
            norm_logits = self.lm_head(decoder_outputs.last_hidden_state)
            outputs["norm_logits"] = norm_logits

            if labels is not None:
                norm_loss = F.cross_entropy(
                    norm_logits.view(-1, norm_logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
                outputs["norm_loss"] = norm_loss

        return outputs

    @torch.no_grad()
    def predict_detection(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Run detection inference. Returns predicted labels (B, S)."""
        encoder_outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        det_logits = self.detection_head(encoder_outputs.last_hidden_state)
        return det_logits.argmax(dim=-1)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **generate_kwargs,
    ) -> torch.Tensor:
        """Run normalization generation. Wraps BARTpho generate."""
        # Use the full model's generate, which handles encoder internally
        return self.bartpho.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
