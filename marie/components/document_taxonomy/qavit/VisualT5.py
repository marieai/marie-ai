import inspect
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from accelerate.hooks import AlignDevicesHook, add_hook_to_module
from torch import nn
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    T5Config,
    T5ForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutput, ModelOutput, Seq2SeqLMOutput

from .clip_encoder import CLIPVisionTower


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.dummy = nn.Parameter(torch.Tensor([0.0]))
        self.dummy.requires_grad = False  # trick to check device and dtype

    def forward(self, x):
        x = x.to(device=self.dummy.device, dtype=self.dummy.dtype)
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class VisualT5(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.vision = None
        self.translation_mlp = None
        self.vit_type = None
        self.instruction_embeddings = None
        self.instruction_tokenizer = None
        self.instruction_embedder = None
        self.instruction_dim = self.model_dim

    def initialize_vision_modules(self, config):
        if config['vit_type'] == 'qavit':
            self.instruction_embedder = self.base_model.shared
            self.instruction_dim = self.instruction_embedder.weight.shape[-1]
        self.vision = CLIPVisionTower(
            config=config, instruction_dim=self.instruction_dim
        )
        self.vit_type = config['vit_type']
        # Vision-> Language projection module
        self.translation_mlp = None
        if config['hidden_translation_layers'] > 0:
            self.translation_mlp = MLP(
                self.vision.hidden_size,
                self.model_dim,
                self.model_dim,
                config['hidden_translation_layers'],
            )

    def freeze_llm_weights(self, config):
        if config['freeze_llm']:
            for param in self.parameters():
                param.requires_grad = False

    def freeze_vision_weights(self, config):
        if config['freeze_clip'] and self.vision is not None:
            self.freeze_clip = True
            for name, param in self.vision.named_parameters():
                if (
                    'instruct' not in name
                ):  # qa-vit components are named with instruct and are trainables
                    param.requires_grad = False

    def get_image_processor(self):
        return self.vision.image_processor

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.config._name_or_path)

    def forward(
        self,
        images=None,
        n=None,
        weights=None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        instructions_list=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            input_embeds = self.encoder.embed_tokens(
                input_ids.to(self.encoder.embed_tokens.weight.device)
            )
            instruct_features = None
            instruct_masks = attention_mask
            # Embed Instruction, if needed
            if self.instruction_embedder is not None:
                outs = self.get_tokenizer()(
                    instructions_list,
                    return_tensors="pt",
                    padding='longest',
                    truncation=True,
                    max_length=512,
                ).to(attention_mask.device)
                instruction_tokens = outs.input_ids
                instruct_masks = outs.attention_mask
                instruct_embeds = self.instruction_embedder(
                    instruction_tokens.to(self.encoder.embed_tokens.weight.device)
                )
                instruct_embeds = self.encoder(
                    inputs_embeds=instruct_embeds, attention_mask=instruct_masks
                ).last_hidden_state
                instruct_features = instruct_embeds
            # Encode vision
            # enable grad if ViT is unfrozen or when we apply qavit
            with torch.set_grad_enabled(
                (not self.freeze_clip) or self.vit_type == 'qavit'
            ):
                image_features = self.vision(
                    instruct_states=instruct_features,
                    instruct_masks=instruct_masks,
                    **images,
                )
                image_atts = torch.ones(
                    image_features.size()[:-1],
                    dtype=torch.long,
                    device=image_features.device,
                )
            image_features = self.translation_mlp(image_features)
            # Concat image + text features
            input_embeds = torch.cat([image_features, input_embeds], dim=1)
            attention_mask = torch.cat([image_atts, attention_mask], dim=1)
            # Encode in LLM
            encoder_outputs = self.encoder(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        hidden_states = encoder_outputs[0]

        # Align to the case of multiple answers per question
        if n is not None:
            extended_encoder_features = []
            extended_input_attn_mask = []
            n = n.int().tolist()
            for b, n_i in enumerate(n):
                extended_encoder_features += [hidden_states[b]] * n_i
                extended_input_attn_mask += [attention_mask[b]] * n_i
            hidden_states = torch.stack(extended_encoder_features, 0)
            attention_mask = torch.stack(extended_input_attn_mask, 0)
        #

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device
                )

        # Decode in LLM
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            loss = loss.view(lm_logits.size(0), -1).mean(1)
            loss = weights * loss
            loss = loss.sum() / labels.size(0)
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def _prepare_encoder_decoder_kwargs_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs,
        model_input_name: Optional[str],
        generation_config: GenerationConfig,
    ) -> Dict[str, Any]:

        # 0. Prepare model kwargs
        instructions_list = model_kwargs.pop("instructions_list", None)

        # 1. get encoder
        encoder = self.get_encoder()
        # Compatibility with Accelerate big model inference: we need the encoder to outputs stuff on the same device
        # as the inputs.
        if hasattr(self, "hf_device_map"):
            if hasattr(encoder, "_hf_hook"):
                encoder._hf_hook.io_same_device = True
            else:
                add_hook_to_module(encoder, AlignDevicesHook(io_same_device=True))

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = (
            "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        )
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value
                for argument, value in encoder_kwargs.items()
                if argument in encoder_signature
            }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = (
            model_input_name if model_input_name is not None else self.main_input_name
        )
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        #
        input_embeds = encoder.embed_tokens(encoder_kwargs['input_ids'])
        # Encode Vision and Text
        # Encode vision
        if self.vision is not None:
            instruct_features = None
            instruct_masks = encoder_kwargs['attention_mask'].clone().detach()
            # Embed Instruction, if needed
            if self.instruction_embedder is not None:
                outs = self.get_tokenizer()(
                    instructions_list,
                    return_tensors="pt",
                    padding='longest',
                    truncation=True,
                    max_length=512,
                ).to(encoder_kwargs['attention_mask'].device)
                instruction_tokens = outs.input_ids
                instruct_masks = outs.attention_mask
                instruct_embeds = self.instruction_embedder(
                    instruction_tokens.to(self.encoder.embed_tokens.weight.device)
                )
                instruct_embeds = self.encoder(
                    inputs_embeds=instruct_embeds, attention_mask=instruct_masks
                ).last_hidden_state
                instruct_features = instruct_embeds
            image_features = self.vision(
                instruct_states=instruct_features,
                instruct_masks=instruct_masks,
                **model_kwargs['images'],
            )
            image_atts = torch.ones(image_features.size()[:-1], dtype=torch.long).to(
                image_features.device
            )

            if self.translation_mlp is not None:
                image_features = self.translation_mlp(image_features)
            # Concat image + text
            input_embeds = torch.cat([image_features, input_embeds], dim=1)
            encoder_kwargs['attention_mask'] = torch.cat(
                [image_atts, encoder_kwargs['attention_mask']], dim=1
            )
        #
        encoder_kwargs['inputs_embeds'] = input_embeds
        del encoder_kwargs['input_ids']
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)
        model_kwargs["attention_mask"] = encoder_kwargs['attention_mask']
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }
