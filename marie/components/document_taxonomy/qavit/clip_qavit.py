from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import einsum, nn
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.clip.configuration_clip import (
    CLIPConfig,
    CLIPTextConfig,
    CLIPVisionConfig,
)
from transformers.models.clip.modeling_clip import (
    CLIPMLP,
    CLIPAttention,
    CLIPEncoderLayer,
    CLIPPreTrainedModel,
    CLIPVisionEmbeddings,
)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.dummy = nn.Parameter(torch.Tensor([0.0]))
        self.dummy.requires_grad = False  # trick to check device and dtype
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, x):
        x = x.to(device=self.dummy.device, dtype=self.dummy.dtype)
        x = self.ln(x)
        for i, layer in enumerate(self.layers):
            x = F.gelu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def exists(val):
    return val is not None


def FeedForward(in_dim, out_dim, inner_dim=None):
    if inner_dim is None:
        inner_dim = out_dim
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, inner_dim, bias=False),
        nn.GELU(),
        # nn.LayerNorm(in_dim),
        nn.Linear(inner_dim, out_dim, bias=False),
        # nn.LayerNorm(in_dim),
    )


class MMCLIPAttention(CLIPAttention):
    def __init__(self, config):
        super().__init__(config)
        self.instruction_gate = nn.Parameter(torch.Tensor([0.0]))
        (
            self.instruction_out_proj,
            self.instruct_q_proj,
            self.instruct_v_proj,
            self.instruct_k_proj,
        ) = (None,) * 4
        self.instruction_out_proj = torch.nn.Linear(
            self.out_proj.in_features, self.out_proj.out_features
        )
        self.instruction_proj_gate = nn.Parameter(torch.Tensor([0.0]))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        kv_states: torch.Tensor = None,
        kv_masks: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        mm_len = kv_states.shape[1]
        hidden_states = torch.cat([kv_states, hidden_states], dim=1)
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        attn_weights = attn_weights[:, mm_len:, :]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        attn_output = torch.bmm(attn_probs, value_states)

        tgt_len -= mm_len
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        gate_val = self.instruction_proj_gate.tanh()
        attn_output = self.out_proj(attn_output) + (
            self.instruction_out_proj(attn_output) * gate_val
        )

        return attn_output, attn_weights_reshaped


class MMCLIPEncoderLayer(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = MMCLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        #
        self.instruct_dim_reduce = FeedForward(
            config.instruction_dim, config.hidden_size, config.hidden_size
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
        instruct_states: torch.Tensor = None,
        instruct_masks: torch.Tensor = None,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            kv_states=(
                self.instruct_dim_reduce(instruct_states)
                if self.instruct_dim_reduce
                else instruct_states
            ),
            kv_masks=instruct_masks,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class InstructCLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config
        OurCLIPEncoderLayer = MMCLIPEncoderLayer
        modules_list = []
        for layer_id in range(config.num_hidden_layers):
            if config.integration_point == 'all':
                layer = OurCLIPEncoderLayer
            elif config.integration_point == 'early':
                layer = (
                    OurCLIPEncoderLayer
                    if layer_id < (config.num_hidden_layers // 2)
                    else CLIPEncoderLayer
                )
            elif config.integration_point == 'late':
                layer = (
                    CLIPEncoderLayer
                    if layer_id < (config.num_hidden_layers // 2)
                    else OurCLIPEncoderLayer
                )
            elif config.integration_point == 'late2':
                layer = (
                    CLIPEncoderLayer
                    if layer_id < (3 * config.num_hidden_layers // 4)
                    else OurCLIPEncoderLayer
                )
            elif config.integration_point == 'late3':
                layer = (
                    CLIPEncoderLayer
                    if layer_id < (config.num_hidden_layers // 4)
                    else OurCLIPEncoderLayer
                )
            elif config.integration_point == 'sparse':
                layer = OurCLIPEncoderLayer if layer_id % 2 == 0 else CLIPEncoderLayer
            else:
                raise ValueError(f'unsupported {config.insruct_type}')
            modules_list.append(layer(config))
        self.layers = nn.ModuleList(modules_list)
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        #
        instruct_states: torch.Tensor = None,
        instruct_masks: torch.Tensor = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                if isinstance(encoder_layer, CLIPEncoderLayer):
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        causal_attention_mask,
                    )
                else:
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        causal_attention_mask,
                        instruct_states=instruct_states,
                        instruct_masks=instruct_masks,
                    )
            else:
                if isinstance(encoder_layer, CLIPEncoderLayer):
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        causal_attention_mask,
                        output_attentions=output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        causal_attention_mask,
                        output_attentions=output_attentions,
                        instruct_states=instruct_states,
                        instruct_masks=instruct_masks,
                    )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class CLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = InstructCLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        #
        instruct_states: torch.Tensor = None,
        instruct_masks: torch.Tensor = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            instruct_states=instruct_states,
            instruct_masks=instruct_masks,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class InstructCLIPVisionModel(CLIPPreTrainedModel):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: CLIPVisionConfig, instruction_dim, integration_point):
        super().__init__(config)
        self.config.instruction_dim = instruction_dim
        self.config.integration_point = integration_point
        self.config.projection_size = 'base'
        self.vision_model = CLIPVisionTransformer(self.config)
        # Initialize weights and apply final processing
        self.post_init()

    def init_qavit_comps(self):
        with torch.no_grad():
            for layer in range(len(self.base_model.vision_model.encoder.layers)):
                if (
                    hasattr(
                        self.base_model.vision_model.encoder.layers[layer].self_attn,
                        'instruction_out_proj',
                    )
                    and self.base_model.vision_model.encoder.layers[
                        layer
                    ].self_attn.instruction_out_proj
                    is not None
                ):
                    self.base_model.vision_model.encoder.layers[
                        layer
                    ].self_attn.instruction_out_proj.load_state_dict(
                        self.base_model.vision_model.encoder.layers[
                            layer
                        ].self_attn.out_proj.state_dict()
                    )

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        #
        instruct_states: torch.Tensor = None,
        instruct_masks: torch.Tensor = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            instruct_states=instruct_states,
            instruct_masks=instruct_masks,
        )
