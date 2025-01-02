import torch
import torch.nn as nn
from clip_qavit import InstructCLIPVisionModel
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel


class CLIPVisionTower(nn.Module):
    def __init__(self, config, instruction_dim):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = config['vit_model']
        self.select_layer = config['vit_layer']
        self.clip_type = config['vit_type']
        self.instruction_dim = instruction_dim
        self.integration_point = (
            config['integration_point']
            if 'integration_point' in config.keys()
            else None
        )
        self.load_model()

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name
        )
        if self.clip_type == 'qavit':
            # # Supporting different fusions
            config = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
            self.vision_tower = InstructCLIPVisionModel(
                config=config,
                instruction_dim=self.instruction_dim,
                integration_point=self.integration_point,
            )
            pretrained_dict = CLIPVisionModel.from_pretrained(
                self.vision_tower_name
            ).state_dict()
            missing, unexpected = self.vision_tower.load_state_dict(
                pretrained_dict, strict=False
            )
            assert len(unexpected) == 0  # asserts that loading weights was as expected
            self.vision_tower.init_qavit_comps()
        else:
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        image_features = image_features
        return image_features

    def forward(self, pixel_values, **kwargs):
        images = pixel_values
        if type(images) is list:
            raise ValueError(f'pixel_values is expected to be a torch tensor')
        else:
            if self.clip_type == 'qavit':
                image_forward_outs = self.vision_tower(
                    images.to(device=self.device, dtype=self.dtype),
                    instruct_states=kwargs['instruct_states'].to(
                        device=self.device, dtype=self.dtype
                    ),
                    instruct_masks=kwargs['instruct_masks'].to(
                        device=self.device, dtype=self.dtype
                    ),
                    output_hidden_states=True,
                )
                image_features = self.feature_select(image_forward_outs).to(
                    images.dtype
                )
            else:
                image_forward_outs = self.vision_tower(
                    images.to(device=self.device, dtype=self.dtype),
                    output_hidden_states=True,
                )
                image_features = self.feature_select(image_forward_outs).to(
                    images.dtype
                )

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
