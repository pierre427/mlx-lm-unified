# Copyright © 2026 Apple Inc.

from dataclasses import dataclass

from . import qwen3_next
from .base import BaseModelArgs
from .qwen3_5 import Model as Qwen3_5Model
from .qwen3_next import transform_moe_weights


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


class Model(Qwen3_5Model):

    def sanitize(self, weights):
        new_weights = {}
        for key, value in weights.items():
            if key.startswith("vision_tower") or key.startswith("model.visual"):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif key.startswith("language_model."):
                pass
            else:
                key = "language_model." + key
            new_weights[key] = value

        args = self.language_model.args
        mlp_prefixes = [
            f"language_model.model.layers.{l}.mlp"
            for l in range(args.num_hidden_layers)
        ]
        if getattr(self.language_model, "mtp", None) is not None:
            mlp_prefixes.extend(
                f"language_model.mtp.layers.{l}.mlp"
                for l in range(getattr(args, "mtp_num_hidden_layers", 0) or 0)
            )

        for prefix in mlp_prefixes:
            gate_up_key = f"{prefix}.experts.gate_up_proj"
            if gate_up_key not in new_weights:
                continue
            gate_up = new_weights.pop(gate_up_key)
            if qwen3_next._MOE_FUSED_GATE_UP:
                # The fused lever consumes the shipped [gate|up] layout as is.
                new_weights[f"{prefix}.switch_mlp.gate_up_proj.weight"] = gate_up
            else:
                mid = gate_up.shape[-2] // 2
                new_weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[
                    ..., :mid, :
                ]
                new_weights[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[
                    ..., mid:, :
                ]
            new_weights[f"{prefix}.switch_mlp.down_proj.weight"] = new_weights.pop(
                f"{prefix}.experts.down_proj"
            )

        # MLX checkpoints ship separate switch_mlp.gate_proj / up_proj tables.
        # The same SparseMoeBlock as qwen4_exp builds gate_up_proj under the
        # fused lever, so apply the same load-time transform here.
        transform_moe_weights(
            new_weights,
            mlp_prefixes,
            fuse_gate_up=qwen3_next._MOE_FUSED_GATE_UP,
            fold_shared=qwen3_next._MOE_SHARED_IN_GATHER,
        )
        return self.language_model.sanitize(new_weights)
