import nodes
import comfy.utils
import comfy.model_management
import comfy.clip_vision
from comfy_api.latest import io
import node_helpers
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import nodes

from comfy import model_management
import comfy.nested_tensor

# Ensure these remain at the top for dimension handling
import torch.nn.functional as F

import comfy.model_management
import comfy.nested_tensor
import comfy.samplers
# We import the Guider class directly from the core
from comfy.samplers import CFGGuider
# The critical missing import for 2026 builds:
from comfy.model_patcher import ModelPatcher 

# --- AUDIO NODES ---

class IDLoRAAudioPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_sample_rate": ("INT", {"default": 24000}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "preprocess"
    CATEGORY = "ID-LoRA/Audio"

    def preprocess(self, audio, target_sample_rate):
        waveform = audio["waveform"]
        sr = audio["sample_rate"]

        if waveform.shape[1] > 1: # Convert to Mono
            waveform = torch.mean(waveform, dim=1, keepdim=True)

        if sr != target_sample_rate: # Resample to 24kHz
            resampler = T.Resample(sr, target_sample_rate)
            waveform = resampler(waveform)

        max_samples = target_sample_rate * 5 # 5 seconds cap
        if waveform.shape[-1] > max_samples:
            waveform = waveform[..., :max_samples]

        return ({"waveform": waveform, "sample_rate": target_sample_rate},)

class IDLoRAAudioVAEEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "audio_vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("audio_latent",)
    FUNCTION = "encode"
    CATEGORY = "ID-LoRA/Audio"

    def encode(self, audio, audio_vae):
        # 1. Extract waveform tensor safely from the input dictionary
        # Native nodes receive the raw dictionary/tensor without wrapper 'boxing'
        if isinstance(audio, dict):
            actual_tensor = audio.get("waveform")
            input_rate = audio.get("sample_rate", 24000)
        else:
            actual_tensor = audio
            input_rate = 24000

        # 2. Prepare the payload for the VAE
        vae_input_dict = {"waveform": actual_tensor, "sample_rate": input_rate}
        
        # 3. Perform encoding
        audio_latents = audio_vae.encode(vae_input_dict) 

        # 4. Ensure 5D for ID-LoRA Joint Latent handling
        # Core expects [Batch, Channels, Temporal, Height, Width]
        if audio_latents.dim() == 3: 
            audio_latents = audio_latents.unsqueeze(-1).unsqueeze(-1)

        # 5. Get sample rate from the VAE model attributes
        if hasattr(audio_vae, "vae_model"):
            s_rate = int(audio_vae.vae_model.sample_rate)
        else:
            s_rate = int(getattr(audio_vae, "sample_rate", 24000))

        # 6. Return a standard ComfyUI Latent dictionary
        return ({"samples": audio_latents, "sample_rate": s_rate, "type": "audio"},)


class IDLoRAAudioVAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_latent": ("LATENT",),
                "audio_vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "decode"
    CATEGORY = "ID-LoRA/Audio"

    def decode(self, audio_latent, audio_vae):
        # 1. Get the raw samples tensor
        samples = audio_latent["samples"]

        # --- THE NESTED TENSOR FIX ---
        # This converts the memory-optimized 'NestedTensor' back to a standard 
        # 'Dense' tensor that the Tiled Scale/Narrow functions can understand.
        if hasattr(samples, "is_nested") and samples.is_nested:
            samples = samples.to_padded_tensor(0.0)
        # -----------------------------

        # 2. Handle Dimension Squeezing
        # ID-LoRA often uses 5D [B, C, T, 1, 1], but VAEs usually want 3D [B, C, T]
        if samples.dim() == 5:
            samples = samples.squeeze(-1).squeeze(-1)
        
        # 3. Perform the actual decoding
        latents_to_decode = {"samples": samples}
        
        # We use a try/except here because different Audio VAEs 
        # have different internal requirements for the dict keys
        try:
            decoded_audio = audio_vae.decode(latents_to_decode)
        except Exception as e:
            # Fallback for VAEs that just want the raw tensor
            decoded_audio = audio_vae.decode(samples)

        # 4. Ensure output is a standard ComfyUI Audio dictionary
        if isinstance(decoded_audio, torch.Tensor):
            sr = getattr(audio_vae, "sample_rate", 24000)
            # FORCE TO CPU HERE
            return ({"waveform": decoded_audio.detach().cpu(), "sample_rate": sr},)
        
        # If the VAE returned a dict, we still need to ensure the waveform inside is CPU
        if isinstance(decoded_audio, dict) and "waveform" in decoded_audio:
            decoded_audio["waveform"] = decoded_audio["waveform"].detach().cpu()
        
        return (decoded_audio,)
# --- LATENT NODES ---

class IDLoRAEmptyAudioLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_vae": ("VAE",),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "seconds": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 30.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("audio_latent",)
    FUNCTION = "generate"
    CATEGORY = "ID-LoRA/Audio"

    def generate(self, audio_vae, batch_size, seconds):
        # 1. Safely determine latent channels and sample rate
        # Native access to the VAE object ensures we don't hit wrapper 'NoneType' errors
        if hasattr(audio_vae, "vae_model"):
            latent_channels = getattr(audio_vae.vae_model, "latent_channels", 4)
            sample_rate = getattr(audio_vae.vae_model, "sample_rate", 24000)
        else:
            latent_channels = getattr(audio_vae, "latent_channels", 4)
            sample_rate = getattr(audio_vae, "sample_rate", 24000)

        # 2. Calculate temporal dimension
        # (seconds * sample_rate) / compression_ratio (usually 1 or based on VAE)
        # For most ID-LoRA Audio VAEs, we aim for the encoded sequence length
        compression_ratio = 1 # Adjust if your specific VAE uses a downsampling factor
        t_dim = int((seconds * sample_rate) / compression_ratio)

        # 3. Create the empty tensor [Batch, Channels, Temporal, Height, Width]
        # Audio latents are typically 1x1 in spatial dimensions (H, W)
        samples = torch.zeros([batch_size, latent_channels, t_dim, 1, 1])

        # 4. Return as a standard ComfyUI Latent dictionary
        return ({"samples": samples, "sample_rate": sample_rate, "type": "audio"},)

class IDLoRAPrepareJointLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_latent": ("LATENT",),
                "audio_latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "prepare"
    CATEGORY = "ID-LoRA/Latent"

    def prepare(self, video_latent, audio_latent):
        v_samples = video_latent["samples"]
        a_samples = audio_latent["samples"]
        
        # We wrap both into a NestedTensor to handle the different 4D/5D shapes
        # correctly during sampling
        joint_samples = comfy.nested_tensor.NestedTensor((v_samples, a_samples))
        
        # Build the output latent dictionary
        output = {"samples": joint_samples}
        
        # Carry over metadata from the video latent (batch, width, height, etc.)
        for key in video_latent:
            if key not in ["samples", "noise_mask"]:
                output[key] = video_latent[key]
                
        return (output,)


# --- CONDITIONING & GUIDING ---

class IDLoRAConditioningSetAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "audio_latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "set_audio"
    CATEGORY = "ID-LoRA/Conditioning"

    def set_audio(self, conditioning, audio_latent):
        # 1. Extract the raw audio tensor from the latent dictionary
        actual_audio_tensor = audio_latent
        if isinstance(audio_latent, dict) and "samples" in audio_latent:
            actual_audio_tensor = audio_latent["samples"]

        new_conditioning = []
        
        # 2. Process the conditioning list natively
        # Native conditioning is a list of [Tensor, {"metadata_key": value}]
        for t in conditioning:
            if isinstance(t, (list, tuple)) and len(t) >= 2:
                # Clone the tensor to avoid modifying other nodes' data
                tensor_part = t[0].clone() if hasattr(t[0], "clone") else t[0]
                metadata_part = t[1]

                # Ensure metadata is a dictionary and copy it
                if isinstance(metadata_part, dict):
                    m = metadata_part.copy()
                else:
                    m = {}

                # Inject the audio latent for the ID-LoRA model to find later
                m["audio_latent"] = actual_audio_tensor
                new_conditioning.append([tensor_part, m])
            else:
                # If it's not a standard pair, pass it through to avoid breaking the pipe
                new_conditioning.append(t)
        
        # 3. Return as a standard ComfyUI tuple
        return (new_conditioning,)


class IDLoRAGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "negative_cond": ("CONDITIONING",),
                "latent_video": ("LATENT",),
                "first_frame_ref": ("LATENT",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "id_cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "audio_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                "audio_latent": ("LATENT",),
                "id_dropout_cond": ("CONDITIONING",),
            }
        }

    # IMPORTANT: Updated to return two types
    RETURN_TYPES = ("GUIDER", "LATENT")
    RETURN_NAMES = ("guider", "combined_latent")
    FUNCTION = "setup_guider"
    CATEGORY = "ID-LoRA/Sampling"

    def setup_guider(self, model, conditioning, negative_cond, latent_video, first_frame_ref, cfg, strength, id_cfg, audio_cfg, audio_latent=None, id_dropout_cond=None):
        import comfy.samplers
        import torch

        # --- 1. ROBUST LATENT HANDLING ---
        samples = latent_video.get("samples")
        
        if hasattr(samples, "is_nested") and samples.is_nested:
            unbound_samples = samples.unbind()
            valid_latents = [t for t in unbound_samples if len(t.shape) == 5]
            latents = valid_latents[0].clone() if valid_latents else unbound_samples[0].clone()
        else:
            latents = torch.as_tensor(samples).clone()

        if latents.dim() == 4:
            latents = latents.unsqueeze(0) 
        
        B, C, F, H, W = latents.shape
        ref = first_frame_ref.get("samples").clone()

        # --- 1.1 NOISE SEEDING ---
        if torch.max(torch.abs(latents)) < 1e-3:
            print("IDLoRA: Seeding Zero Latent with Noise...")
            latents = torch.randn_like(latents) * 1.0

        # --- 2. VISUAL INJECTION ---
        noise_mask = torch.ones((B, 1, F, H, W), device=latents.device, dtype=latents.dtype)
        try:
            if ref.dim() == 5: ref = ref.squeeze(2)
            if latents.shape[-2:] != ref.shape[-2:]:
                ref = torch.nn.functional.interpolate(ref, size=latents.shape[-2:], mode="bilinear")
            if ref.shape[0] != B: ref = ref.repeat(B, 1, 1, 1)

            if ref.shape[1] == C:
                latents[:, :, 0, :, :] = ref * strength + latents[:, :, 0, :, :] * (1.0 - strength)
                noise_mask[:, :, 0, :, :] = 0.1
        except Exception as e:
            print(f"IDLoRA Injection Error: {e}")

        # --- 3. AUDIO PREP (8-Channel Aware) ---
        if audio_latent is not None:
            a_samples = audio_latent.get("samples").clone()
            if a_samples.dim() == 4: a_samples = a_samples.unsqueeze(0)
            
            # LTX Audio VAE typically uses 8 channels
            # If your input audio is 128 (from a previous latent), we take 8.
            # If it's 8, we keep it 8.
            if a_samples.shape[1] > 8:
                a_samples = a_samples[:, :8, ...]
            
            # Now, we need to PADDING this 8-channel audio to 128 
            # so it can be concatenated with the 128-channel Video.
            a_B, a_C, a_F, a_H, a_W = a_samples.shape
            
            # Create 120 channels of padding (8 + 120 = 128)
            padding = torch.zeros((a_B, 128 - a_C, a_F, a_H, a_W), 
                                 device=a_samples.device, 
                                 dtype=a_samples.dtype)
            
            # Place the 8 audio channels at the FRONT
            a_samples_128 = torch.cat([a_samples, padding], dim=1)
            
            # Align Temporal/Spatial (F, H, W) to match Video
            if a_samples_128.shape[2:] != latents.shape[2:]:
                print(f"IDLoRA: Interpolating 8ch Audio to Video Grid {latents.shape[2:]}")
                a_samples_128 = torch.nn.functional.interpolate(
                    a_samples_128, 
                    size=latents.shape[2:], 
                    mode="trilinear", 
                    align_corners=False
                )
            a_samples = a_samples_128 # Re-assign for the rest of the script
        else:
            # Fallback to zeros
            a_samples = torch.zeros_like(latents)

        # --- 4. CONDITIONING METADATA ---
        processed_pos = []
        for t in conditioning:
            if isinstance(t, (list, tuple)) and len(t) >= 2:
                metadata = t[1].copy() if isinstance(t[1], dict) else {}
                metadata["visual_latent"] = ref
                metadata["audio_latent"] = a_samples
                metadata["audio_weight"] = audio_cfg
                processed_pos.append([t[0], metadata])
            else:
                processed_pos.append(t)

        # --- 5. BATCH CONCATENATION ---
        # Video at [0], Audio at [1]
        final_samples = torch.cat([latents, a_samples], dim=0)
        final_mask = torch.cat([noise_mask, torch.ones_like(noise_mask)], dim=0)

        # --- 6. GUIDER SETUP ---
        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(processed_pos, negative_cond)
        guider.set_cfg(cfg) 

        print(f"DEBUG [Guider]: Multimodal Batch Success: {final_samples.shape}")

        return (guider, {"samples": final_samples, "noise_mask": final_mask})

class IDLoRASeparateLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "combined_latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("video_latent", "audio_latent")
    FUNCTION = "separate"
    CATEGORY = "IDLoRA"

    def separate(self, combined_latent):
    import torch
    samples = combined_latent.get("samples") # [2, 128, 16, 26, 20]
    
    # 1. Video stays 5D - take the first batch, all 128 channels
    video_samples = samples[0:1, ...] # [1, 128, 16, 26, 20]
    
    # 2. Audio Latent Correction
    # We take the second batch [1:2], but we MUST slice the channels
    # to only take the first 8 channels that the Audio VAE expects.
    audio_raw = samples[1:2, 0:8, ...] # [1, 8, 16, 26, 20] <--- THE FIX
    
    b, c, f, h, w = audio_raw.shape
    total_length = f * h * w
    
    # LTX Audio VAEs often expect [Batch, Channels, Length, 1] 
    # or [Batch, Channels, Height, Width]
    # Given your error expected [1, 8, 8322, 3], let's match that spatial target
    audio_samples = audio_raw.reshape(b, c, total_length, 1) 

    video_out = {"samples": video_samples}
    audio_out = {"samples": audio_samples}

    if "noise_mask" in combined_latent:
        mask = combined_latent["noise_mask"]
        if mask is not None and torch.is_tensor(mask):
            video_out["noise_mask"] = mask[0:1, ...]
    
    return (video_out, audio_out)


class IDLoRAPromptFormatter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id_tag": ("STRING", {
                    "default": "[NAME]",
                    "label": "Identity Tag (e.g., [NAME])"
                }),
                "subject_description": ("STRING", {
                    "multiline": True, 
                    "default": "a middle-aged man with a kind face",
                    "label": "VISUAL: Character Description"
                }),
                "visual_action": ("STRING", {
                    "multiline": True, 
                    "default": "nodding slowly while looking at the camera",
                    "label": "VISUAL: Movement & Action"
                }),
                "dialogue_text": ("STRING", {
                    "multiline": True, 
                    "default": "I think we've finally found the solution.",
                    "label": "SPEECH: Dialogue Content"
                }),
                "environmental_sounds": ("STRING", {
                    "multiline": True, 
                    "default": "soft rain pattering against a window, distant thunder",
                    "label": "SOUNDS: Environment Audio"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True, 
                    "default": "low quality, blurry, static, digital noise, out of sync, distorted voice, muffled audio, background music, watermark, text",
                    "label": "NEGATIVE: Exclude these qualities"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "format_prompt"
    CATEGORY = "ID-LoRA/Prompting"

    def format_prompt(self, id_tag, subject_description, visual_action, dialogue_text, environmental_sounds, negative_prompt):
        # LTX-2.3 / ID-LoRA Multi-Modal Prompt Construction
        # [VISUAL] targets the DiT video stream
        # [SPEECH] targets the Lip-Sync/Audio-Visual binding
        # [SOUNDS] targets the latent audio vocoder
        
        pos = (
            f"[VISUAL]: {id_tag} {subject_description}, {visual_action}. "
            f"[SPEECH]: {id_tag} says \"{dialogue_text}\". "
            f"[SOUNDS]: {environmental_sounds}."
        )
        
        return (pos, negative_prompt)

NODE_CLASS_MAPPINGS = {
    "IDLoRAGuider": IDLoRAGuider,
    "IDLoRAAudioPreprocessor": IDLoRAAudioPreprocessor,
    "IDLoRAConditioningSetAudio": IDLoRAConditioningSetAudio,
    "IDLoRAAudioVAEEncode": IDLoRAAudioVAEEncode,
    "IDLoRAPromptFormatter": IDLoRAPromptFormatter,
    "IDLoRASeparateLatent": IDLoRASeparateLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IDLoRAGuider": "ID-LoRA Guider",
}
