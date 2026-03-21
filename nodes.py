import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import comfy.samplers

# --- AUDIO PREPARATION ---

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
        if waveform.shape[1] > 1: 
            waveform = torch.mean(waveform, dim=1, keepdim=True)
        if sr != target_sample_rate:
            resampler = T.Resample(sr, target_sample_rate)
            waveform = resampler(waveform)
        return ({"waveform": waveform, "sample_rate": target_sample_rate},)

class IDLoRAAudioVAEEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"audio": ("AUDIO",), "audio_vae": ("VAE",)}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "ID-LoRA/Audio"

    def encode(self, audio, audio_vae):
        vae_input = {"waveform": audio["waveform"], "sample_rate": audio["sample_rate"]}
        latents = audio_vae.encode(vae_input)
        if latents.dim() == 3: # Ensure 5D [B, C, T, 1, 1]
            latents = latents.unsqueeze(-1).unsqueeze(-1)
        return ({"samples": latents, "sample_rate": audio["sample_rate"], "type": "audio"},)

# --- VIDEO PREPARATION (First Frame Injection) ---

class IDLoRAPrepareVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_latent": ("LATENT",),
                "first_frame_ref": ("LATENT",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "prepare"
    CATEGORY = "ID-LoRA/Video"

    def prepare(self, video_latent, first_frame_ref, strength):
        v = video_latent["samples"].clone()
        ref = first_frame_ref["samples"].clone()
        if ref.dim() == 5: ref = ref.squeeze(2)
        if v.shape[-2:] != ref.shape[-2:]:
            ref = F.interpolate(ref, size=v.shape[-2:], mode="bilinear")
        v[:, :, 0, :, :] = ref * strength + v[:, :, 0, :, :] * (1.0 - strength)
        return ({"samples": v},)

# --- ID-LORA CONDITIONING & GUIDER ---

class IDLoRAConditioningSetAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING",), "audio_latent": ("LATENT",)}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "set_audio"
    CATEGORY = "ID-LoRA/Conditioning"

    def set_audio(self, conditioning, audio_latent):
        actual_audio_tensor = audio_latent["samples"]
        new_conditioning = []
        for t in conditioning:
            m = t[1].copy()
            m["audio_latent"] = actual_audio_tensor # The "Identity" Reference
            new_conditioning.append([t[0].clone(), m])
        return (new_conditioning,)

class IDLoRAGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "negative_cond": ("CONDITIONING",),
                "joint_latent": ("LATENT",),
                "cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1, "label": "Global CFG"}),
                "id_cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1, "label": "Identity CFG"}),
                "audio_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1, "label": "Audio CFG"}),
            },
            "optional": {
                "id_dropout_cond": ("CONDITIONING",), # Standard text-only conditioning
            }
        }

    RETURN_TYPES = ("GUIDER", "LATENT")
    RETURN_NAMES = ("guider", "joint_latent")
    FUNCTION = "setup"
    CATEGORY = "ID-LoRA/Sampling"

    def setup(self, model, conditioning, negative_cond, joint_latent, cfg, id_cfg, audio_cfg, id_dropout_cond=None):
        processed_pos = []
        
        for t in conditioning:
            m = t[1].copy()
            # Inject the specialized weights for the ID-LoRA DiT layers
            m["id_weight"] = id_cfg      # Identity/Speaker strength
            m["audio_weight"] = audio_cfg # Environment/Audio sync strength
            
            # DROPOUT LOGIC: 
            # If a dropout condition (text-only) is provided, we attach it to the 
            # metadata. The sampler uses this to calculate the 'Identity Delta'.
            if id_dropout_cond is not None:
                # We take the first conditioning tensor from the dropout list
                m["id_dropout_samples"] = id_dropout_cond[0][0]
            
            processed_pos.append([t[0].clone(), m])

        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(processed_pos, negative_cond)
        guider.set_cfg(cfg) 
        
        return (guider, joint_latent)

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
    "IDLoRAAudioPreprocessor": IDLoRAAudioPreprocessor,
    "IDLoRAAudioVAEEncode": IDLoRAAudioVAEEncode,
    "IDLoRAPrepareVideo": IDLoRAPrepareVideo,
    "IDLoRAConditioningSetAudio": IDLoRAConditioningSetAudio,
    "IDLoRAGuider": IDLoRAGuider,
    "IDLoRAPromptFormatter": IDLoRAPromptFormatter,
}

NODE_DISPLAY_NAME_MAPPINGS = {}
