"""
BLIP-2 Models

Hierarchy:
- Blip2Base: Base model with vision encoder + Q-Former
  └── Blip2Qformer: Stage 1 (adds ITC, ITM, ITG losses)
      └── Blip2OPT: Stage 2 MAIN MODEL (adds LLM + Projector + LM loss)

Blip2OPT.forward() returns: loss_itc + loss_itm + loss_itg + loss_lm
"""

from .blip2 import Blip2Base
from .blip2_qformer import Blip2Qformer
from .blip2_opt import Blip2OPT

__all__ = ["Blip2Base", "Blip2Qformer", "Blip2OPT"]
