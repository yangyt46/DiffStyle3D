import torch
import torch.nn as nn
from diffusers import (
    AutoencoderKLTemporalDecoder,
    DDIMScheduler,
)
from stable_diffusion.pipeline_SD import StableDiffusionWarpPipeline

import torch.nn.functional as F


def seed_all_rngs(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class AttentionLayerTracker:
    def __init__(self, layer_range=(0, 16)):
        self.start_layer, self.end_layer = layer_range
        self.self_layers = list(range(self.start_layer, self.end_layer))
        self.num_self_layers = -1
        self.current_self_layer = 0

    def reset(self):
        self.current_self_layer = 0

    def should_record(self):
        return self.current_self_layer in self.self_layers

    def advance(self):
        self.current_self_layer += 1

class StyleDiffusion(nn.Module):
    def __init__(self, style_image, device, start_layer, end_layer, guidance_opt, height=512, width=512,
                 num_inference_steps=50, attention_scale=1, content_weight=0.15, fp16=False):
        super().__init__()
        seed_all_rngs(3407)
        self.dtype = torch.float16 if fp16 else torch.float32
        self.device = device
        self.scheduler = DDIMScheduler.from_pretrained(guidance_opt.sd_model_key, subfolder="scheduler")
        if guidance_opt.svd_vae:
            vae = AutoencoderKLTemporalDecoder.from_pretrained(guidance_opt.sd_model_key,
                                                               subfolder="svd-vae")
            self.pipeline = StableDiffusionWarpPipeline.from_pretrained(
                guidance_opt.sd_model_key, scheduler=self.scheduler, vae=vae, safety_checker=None
            )
        else:
            self.pipeline = StableDiffusionWarpPipeline.from_pretrained(
                guidance_opt.sd_model_key, scheduler=self.scheduler, safety_checker=None
            )
        self.pipeline.to(self.device)
        self.pipeline.enable_xformers_memory_efficient_attention()
        self.unet_model = self.pipeline.unet
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.unet.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        self.unet_model.requires_grad_(False)
        self.layer_tracker = AttentionLayerTracker(layer_range=(start_layer, end_layer))
        self.feature_store = AttentionFeatureStore()
        register_view_consistency_control(
            self.unet_model, layer_tracker=self.layer_tracker, feature_store=self.feature_store
        )
        print("Total self-attention layers in Stable Diffusion: ", self.layer_tracker.num_self_layers)
        print("Layers for extracting self-attention features: ", self.layer_tracker.self_layers)

        self.height = height // self.pipeline.vae_scale_factor
        self.width = width // self.pipeline.vae_scale_factor
        self.num_inference_steps = guidance_opt.num_inference_steps
        self.scheduler.set_timesteps(self.num_inference_steps, device=device)
        self.timesteps = torch.flip(self.scheduler.timesteps, dims=(0,))
        self.attention_scale = attention_scale
        self.style_latent = self.encode_image_to_latent(style_image)
        self.content_weight = content_weight
        self.batch_size = guidance_opt.batch_size
        self.add_noise = guidance_opt.add_noise

        #############  init
        self.alphas = self.pipeline.scheduler.alphas_cumprod.to(self.device)
        self.null_embeds = self.pipeline.encode_prompt("", self.device, 1, False)[0]
        self.null_embeds_for_latents = self.null_embeds.repeat(self.batch_size, 1, 1)
        self.null_embeds_for_style = self.null_embeds.repeat(self.style_latent.shape[0], 1, 1)
        self.null_embeds_for_content = self.null_embeds.repeat(self.batch_size, 1, 1)

        self.selected_timestep = self.timesteps[guidance_opt.set_timestep]
        print("Select Timestep:",self.selected_timestep)
        with torch.no_grad():
            self.style_queries, self.style_keys, self.style_values, self.style_outputs = self.extract_attention_features(
                self.style_latent,
                self.selected_timestep.repeat(self.style_latent.size(0)),
                self.null_embeds_for_style,
                add_noise=self.add_noise,
            )
    @torch.no_grad()
    def encode_image_to_latent(self, image):
        dtype = next(self.pipeline.vae.parameters()).dtype
        image = image.to(device=self.device, dtype=dtype) * 2.0 - 1.0
        latent = self.pipeline.vae.encode(image)["latent_dist"].mean
        latent = latent * self.pipeline.vae.config.scaling_factor
        return latent

    def encode_image_to_trainable_latent(self, image):
        dtype = next(self.pipeline.vae.parameters()).dtype
        image = image.to(device=self.device, dtype=dtype) * 2.0 - 1.0
        latent = self.pipeline.vae.encode(image)["latent_dist"].mean
        latent = latent * self.pipeline.vae.config.scaling_factor
        latent = torch.clamp(latent, -4.0, 4.0)
        return latent

    def resize_warp_inputs(
            self,
            mask,
            grid,
            visible
    ):
        height_list = [self.height, self.height // 2, self.height // 4, self.height // 8]
        mask_dict = {}
        visible_dict = {}
        grid_dict = {}
        batch_size, view_count, height, width, _ = grid.shape
        grids_reshaped = grid.view(batch_size * view_count, height, width, 2)
        for target_height in height_list:
            mask_rescaled = F.interpolate(
                mask.unsqueeze(1),
                size=(target_height, target_height),
                mode='bilinear'
            ).squeeze(1)
            mask_dict[target_height] = (mask_rescaled > 0.5).float()
            visible_dict[target_height] = F.interpolate(
                visible,
                size=(target_height, target_height),
                mode='bilinear'
            )
            grids_resized = F.interpolate(
                grids_reshaped.permute(0, 3, 1, 2),
                size=(target_height, target_height),
                mode="bilinear",
                align_corners=True
            ).permute(0, 2, 3, 1)
            grids_resized = grids_resized.view(
                batch_size, view_count, target_height, target_height, 2
            )
            grid_dict[target_height] = grids_resized
        return mask_dict, grid_dict, visible_dict

    def train_style(
            self,
            latents,
            iteration,
            gt_content=None,
            grid=None,
            mask=None,
            visible=None,
    ):
        mask_list, grid_list, visible_list = self.resize_warp_inputs(mask, grid, visible)
        content_latent = self.encode_image_to_latent(gt_content)
        latents = self.encode_image_to_trainable_latent(latents)

        with torch.no_grad():
            _content_queries, _content_keys, _content_values, content_outputs = self.extract_attention_features(
                content_latent,
                self.selected_timestep,
                self.null_embeds_for_content,
                add_noise=self.add_noise,
            )
        current_queries, _current_keys, _current_values, current_outputs = self.extract_attention_features(
            latents,
            self.selected_timestep,
            self.null_embeds_for_latents,
            grid_list=grid_list,
            visible_list=visible_list,
            add_noise=self.add_noise,
        )
        style_loss = self.compute_style_loss(
            current_queries,
            self.style_keys,
            self.style_values,
            current_outputs,
            scale=self.attention_scale,
            mask_list=mask_list,
        )
        content_loss = self.compute_content_loss(current_outputs, content_outputs, mask_list=mask_list)
        Attention_Aware_Loss = style_loss + content_loss * self.content_weight
        return Attention_Aware_Loss


    def extract_attention_features(
            self,
            latent,
            timestep,
            embeds,
            add_noise=False,
            grid_list=None,
            visible_list=None,
    ):
        self.feature_store.clear()
        self.layer_tracker.reset()
        if add_noise:
            noise = torch.randn_like(latent)
            noisy_latent = self.scheduler.add_noise(latent, noise, timestep)
        else:
            noisy_latent = latent
        if grid_list is not None:
            _ = self.unet_model(noisy_latent, timestep, embeds, grid_list=grid_list, visible_list=visible_list)[0]
        else:
            _ = self.unet_model(noisy_latent, timestep, embeds)[0]
        return self.feature_store.snapshot()

    def compute_style_loss(self, query_list, style_key_list, style_value_list, output_list, scale=1,
                           mask_list=None, eps=1e-6):
        loss = 0
        attention_mask = None
        for query, style_keys, style_values, output in zip(query_list, style_key_list, style_value_list, output_list):
            target_out = F.scaled_dot_product_attention(
                query * scale,
                torch.cat(torch.chunk(style_keys, style_keys.shape[0]), 2).repeat(query.shape[0], 1, 1, 1),
                torch.cat(torch.chunk(style_values, style_values.shape[0]), 2).repeat(query.shape[0], 1, 1, 1),
                attn_mask=attention_mask
            )
            height = int(target_out.shape[2] ** 0.5)
            mask = mask_list[height]
            expanded_mask = mask.view(mask.size(0), 1, -1, 1).expand_as(target_out)
            centered_output = output - output.mean(dim=-1, keepdim=True)
            centered_target = target_out - target_out.mean(dim=-1, keepdim=True)
            output_norm = torch.linalg.norm(centered_output, dim=-1, keepdim=True).clamp(min=eps)
            target_norm = torch.linalg.norm(centered_target, dim=-1, keepdim=True).clamp(min=eps)

            normalized_output = centered_output / output_norm
            normalized_target = centered_target / target_norm
            loss += self.masked_mse_loss(normalized_output, normalized_target.detach(), expanded_mask)
        return loss

    def masked_mse_loss(self, prediction, target, mask, eps=1e-6):
        squared_error = (prediction - target) ** 2
        loss = (squared_error * mask).sum() / (mask.sum() + eps)
        return loss

    def compute_content_loss(self, query_list, content_query_list, mask_list=None, eps=1e-6):
        loss = 0
        for query, content_query in zip(query_list, content_query_list):
            height = int(content_query.shape[2] ** 0.5)
            mask = mask_list[height].round().int()
            expanded_mask = mask.view(mask.size(0), 1, -1, 1).expand_as(content_query)
            query_centered = query - query.mean(dim=-1, keepdim=True)
            content_centered = content_query - content_query.mean(dim=-1, keepdim=True)
            query_norm = torch.linalg.norm(query_centered, dim=-1, keepdim=True).clamp(min=eps)
            content_norm = torch.linalg.norm(content_centered, dim=-1, keepdim=True).clamp(min=eps)
            normalized_query = query_centered / query_norm
            normalized_content = content_centered / content_norm
            loss += self.masked_mse_loss(normalized_query, normalized_content.detach(), expanded_mask)
        return loss


class AttentionFeatureStore:
    def __init__(self):
        self.queries = []
        self.keys = []
        self.values = []
        self.outputs = []

    def clear(self):
        self.queries.clear()
        self.keys.clear()
        self.values.clear()
        self.outputs.clear()

    def record(self, query, key, value, output):
        self.queries.append(query)
        self.keys.append(key)
        self.values.append(value)
        self.outputs.append(output)

    def snapshot(self):
        return (
            self.queries.copy(),
            self.keys.copy(),
            self.values.copy(),
            self.outputs.copy(),
        )

def register_view_consistency_control(unet, layer_tracker, feature_store=None):
    def build_Geometry_Guided_Attention_forward(self):
        def forward(
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            grid_list=None,
            visible_list=None,
            *args,
            **kwargs,
        ):
            residual = hidden_states
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(
                    batch_size, channel, height * width
                ).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )
            is_self = encoder_hidden_states is None

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = self.to_q(hidden_states)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
                if grid_list is not None:
                    feature_batch_size, token_count, channel_count = encoder_hidden_states.shape
                    feature_height = int(token_count ** 0.5)
                    features = encoder_hidden_states.view(
                        feature_batch_size, feature_height, feature_height, channel_count
                    ).permute(0, 3, 1, 2)
                    warped_feats = []
                    attention_masks = []
                    for batch_index in range(feature_batch_size):
                        ###############  Explicit Geometry Guidance
                        view_order = [batch_index] + [i for i in range(feature_batch_size) if i != batch_index]
                        source_features = features[view_order]
                        warp_grid = grid_list[feature_height][batch_index][view_order]
                        warped = F.grid_sample(source_features, warp_grid, align_corners=True)
                        warped = warped.permute(0, 2, 3, 1).reshape(
                            1, feature_batch_size * feature_height * feature_height, channel_count
                        )
                        warped_feats.append(warped)
                        visible_mask = visible_list[feature_height][batch_index][view_order].reshape(
                            feature_batch_size * token_count
                        )
                        attention_masks.append(visible_mask)
                    encoder_hidden_states = torch.cat(warped_feats, dim=0)
                    stacked_attention_mask = torch.stack(attention_masks, dim=0)
                    attention_mask = (1.0 - stacked_attention_mask[:, None, None, :]) * -1e4

            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // self.heads

            query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            if is_self and layer_tracker.should_record():
                feature_store.record(query, key, value, hidden_states)

            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, self.heads * head_dim
            )
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(
                    batch_size, channel, height, width
                )
            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            if is_self:
                layer_tracker.advance()

            return hidden_states

        return forward

    def replace_attention_forward(net, count):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == "Attention":  # spatial Transformer layer
                net.forward = build_Geometry_Guided_Attention_forward(net)
                return count + 1
            elif hasattr(net, "children"):
                count = replace_attention_forward(subnet, count)
        return count

    attention_layer_count = 0
    for _, net in unet.named_children():
        attention_layer_count += replace_attention_forward(net, 0)
    layer_tracker.num_self_layers = attention_layer_count // 2
