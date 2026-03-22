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
                "latent_video": ("LATENT",),
                "first_frame_ref": ("LATENT",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "prepare"
    CATEGORY = "ID-LoRA/Video"

    def prepare(self, latent_video, first_frame_ref, strength):
        v = latent_video["samples"].clone()
        ref = first_frame_ref["samples"].clone()
        
        # --- 1. DIMENSION SANITY CHECK ---
        if v.dim() == 4: v = v.unsqueeze(0) # [B, C, F, H, W]
        if ref.dim() == 4: ref = ref.unsqueeze(0)
        
        B, C, F, H, W = v.shape
        
        # --- 2. FIRST FRAME INJECTION (The "Perfect Video" Secret) ---
        try:
            # If ref has a temporal dim, squeeze it to get the single image frame
            if ref.dim() == 5: ref = ref.squeeze(2) 
            
            # Ensure spatial alignment (Match Latent H/W)
            if v.shape[-2:] != ref.shape[-2:]:
                ref = torch.nn.functional.interpolate(ref, size=v.shape[-2:], mode="bilinear")
            
            # Inject the reference into the first frame (Index 0)
            # We blend it based on strength to allow for some flexibility
            v[:, :, 0, :, :] = (ref * strength) + (v[:, :, 0, :, :] * (1.0 - strength))

            # --- 3. THE 'BLACK OUTPUT' FIX ---
            # If the rest of the frames (1 to F) are pure 0.0, the sampler can fail.
            # We fill them with a very faint noise floor if they are empty.
            if torch.max(torch.abs(v[:, :, 1:, :, :])) < 1e-5:
                v[:, :, 1:, :, :] = torch.randn_like(v[:, :, 1:, :, :]) * 0.01            
            
            # --- 4. NOISE MASKING ---
            # Create a mask that tells the sampler: "Don't change frame 0 much"
            mask = torch.ones((B, 1, F, H, W), device=v.device, dtype=v.dtype)
            mask[:, :, 0, :, :] = 0.0  # 0.0 means "Keep this exactly as provided"
            
        except Exception as e:
            print(f"IDLoRA Prepare Error: {e}")
            mask = torch.ones((B, 1, F, H, W), device=v.device, dtype=v.dtype)

        # --- 5. LTXV METADATA ---
        return ({
            "samples": v, 
            "noise_mask": mask,
            "type": "video" # CRITICAL for LTXAV Sampler
        },)

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

class IDLoRAAudioNoiseInjector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",), # Output from the official LTX Empty Audio Node
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "inject_noise"
    CATEGORY = "ID-LoRA/Audio"

    def inject_noise(self, samples, seed, strength):
        s = samples.copy()
        audio_tensor = s["samples"]
        
        # Match the shape exactly to whatever LTX provided
        torch.manual_seed(seed)
        noise = torch.randn_like(audio_tensor)
        
        # If strength is 1.0, it's a pure blank canvas. 
        # If lower, it blends with the input (useful for img2vid style audio)
        s["samples"] = (audio_tensor * (1.0 - strength)) + (noise * strength)
            
        return (s,)


NODE_CLASS_MAPPINGS = {
    "IDLoRAAudioPreprocessor": IDLoRAAudioPreprocessor,
    "IDLoRAAudioVAEEncode": IDLoRAAudioVAEEncode,
    "IDLoRAPrepareVideo": IDLoRAPrepareVideo,
    "IDLoRAConditioningSetAudio": IDLoRAConditioningSetAudio,
    "IDLoRAGuider": IDLoRAGuider,
    "IDLoRAPromptFormatter": IDLoRAPromptFormatter,
    "IDLoRAAudioNoiseInjector": IDLoRAAudioNoiseInjector,
}

NODE_DISPLAY_NAME_MAPPINGS = {}
