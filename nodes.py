import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import comfy.samplers

# --- AUDIO PREPARATION ---

class IDLoRAPrepareAudioReference:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "audio": ("AUDIO",),
                "audio_vae": ("VAE",),
                "target_sample_rate": ("INT", {"default": 24000}),
                "audio_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "process_audio_reference"
    CATEGORY = "ID-LoRA/Conditioning"

    def process_audio_reference(self, conditioning, audio, audio_vae, target_sample_rate, audio_strength):
        # --- 1. PREPROCESS (Resample & Mono) ---
        waveform = audio["waveform"]
        sr = audio["sample_rate"]
        
        # Force Mono
        if waveform.shape[1] > 1:
            waveform = torch.mean(waveform, dim=1, keepdim=True)
            
        # Resample if necessary
        if sr != target_sample_rate:
            resampler = T.Resample(sr, target_sample_rate)
            waveform = resampler(waveform)

        # --- 2. VAE ENCODE ---
        vae_input = {"waveform": waveform, "sample_rate": target_sample_rate}
        latents_dict = audio_vae.encode(vae_input)
        
        # Get the tensor (usually [B, C, T] or [B, C, T, H, W])
        audio_samples = latents_dict["samples"] if isinstance(latents_dict, dict) else latents_dict

        # --- 3. LTX2.3 SCALING & STRENGTH ---
        # Apply the VAE constant (/ 0.18215) and your custom strength multiplier
        audio_samples = (audio_samples / 0.18215) * audio_strength

        # Ensure 5D [B, C, T, 1, 1] for LTX transformer compatibility
        if audio_samples.dim() == 3:
            audio_samples = audio_samples.unsqueeze(-1).unsqueeze(-1)

        # --- 4. CONDITIONING INJECTION ---
        new_conditioning = []
        for t in conditioning:
            # t[0] is the CLIP pooled output, t[1] is the metadata dictionary
            metadata = t[1].copy()
            
            # Inject the processed audio latent as the "Identity Reference"
            metadata["audio_latent"] = audio_samples
            
            # Clone the tensor to avoid side effects
            new_conditioning.append([t[0].clone(), metadata])

        print(f"DEBUG [AudioRef]: Injected Latent with Strength {audio_strength:.2f} | Shape: {audio_samples.shape}")
        
        return (new_conditioning,)

# --- VIDEO PREPARATION (First Frame Injection) ---

class IDLoRAPrepareVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_video": ("LATENT",),
                "first_frame_ref": ("LATENT",),
                "strength": ("FLOAT", {"default": 0.92, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mode": (["Base", "Refiner"], {"default": "Base"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "prepare"
    CATEGORY = "ID-LoRA/Video"

    def prepare(self, latent_video, first_frame_ref, strength, mode):
        v = latent_video["samples"].clone()
        ref = first_frame_ref["samples"].clone()
        
        # 1. Standard LTX2.3 VAE Scaling Correction
        # This is the "Big Fish" for clarity. 
        # LTX expects latents in a specific distribution.
        if torch.std(ref).item() < 0.5:
            ref = ref / 0.18215

        B, C, F, H, W = v.shape
        f_ref_count = ref.shape[2] if ref.dim() == 5 else 1
        
        # 2. KEYFRAME INJECTION (Mode: Base Only)
        # In Refiner mode, we skip injection because we want to refine the 
        # existing video, not overwrite it with a static frame.
        mask = torch.ones((B, 1, F, H, W), device=v.device, dtype=v.dtype)
        
        if mode == "Base":
            for i in range(f_ref_count):
                if i < F:
                    current_f = ref[:, :, i, :, :] if ref.dim() == 5 else ref.squeeze(2)
                    
                    # Ensure spatial matching
                    if v.shape[-2:] != current_f.shape[-2:]:
                        current_f = torch.nn.functional.interpolate(
                            current_f, size=v.shape[-2:], mode="bilinear"
                        )
                    
                    # Inject at the slider strength (e.g., 0.92)
                    v[:, :, i, :, :] = (current_f * strength) + (v[:, :, i, :, :] * (1.0 - strength))
                    
                    # Hard-lock Frame 0 only. 
                    # We leave frames 1-F to be handled by the Scheduler's noise.
                    mask[:, :, i, :, :] = 0.0 

        # 3. OUTPUT
        # For Refiner mode, v remains the original upscaled latent and mask is all 1s.
        return ({"samples": v, "noise_mask": mask, "type": "video"},)

# --- ID-LORA CONDITIONING & GUIDER ---

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
                "id_tag": ("STRING", {"default": "[NAME]"}),
                "primary_desc": ("STRING", {"multiline": True, "default": "a middle-aged man with a kind face"}),
                "secondary_desc": ("STRING", {"multiline": True, "default": ""}), # Leave empty for 1 person
                "visual_action": ("STRING", {"multiline": True, "default": "nodding slowly while looking at the camera"}),
                "dialogue_text": ("STRING", {"multiline": True, "default": "I think we've finally found the solution."}),
                "environmental_sounds": ("STRING", {"multiline": True, "default": "soft rain"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "low quality, blurry..."}),
            }
        }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "format_prompt"
    CATEGORY = "ID-LoRA/Prompting"

    def format_prompt(self, id_tag, primary_desc, secondary_desc, visual_action, dialogue_text, environmental_sounds, negative_prompt):
        # 1. DYNAMIC VISUAL CONSTRUCTION
        if secondary_desc.strip() and secondary_desc.lower() != "none":
            # TWO PERSON MODE: Forces spatial separation to stop speech-hijacking
            visual_block = (
                f"[VISUAL]: {id_tag} {primary_desc} on the left, "
                f"and {secondary_desc} on the right. "
                f"They are {visual_action}."
            )
        else:
            # SINGLE PERSON MODE: Standard centered focus
            visual_block = f"[VISUAL]: {id_tag} {primary_desc}, {visual_action}."

        # 2. DIALOGUE ATTRIBUTION
        # If there are two people, the model needs to see the name 
        # specifically inside the SPEECH tag to link the audio to the correct mouth.
        speech_block = f"[SPEECH]: {dialogue_text}"
        
        # 3. SOUNDS
        sound_block = f"[SOUNDS]: {environmental_sounds}."

        pos = f"{visual_block} {speech_block} {sound_block}"
        
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
        
        # 1. Scaling for LTX2.3 (The VAE Constant)
        # This keeps the audio latents in the same numerical 'range' as the video
        #scaling_factor = 0.18215
        
        torch.manual_seed(seed)
        noise = torch.randn_like(audio_tensor)
        
        # 2. Blending
        # Using strength 1.0 here means we are starting with 'Balanced LTX Noise'
        s["samples"] = (audio_tensor * (1.0 - strength)) + (noise * strength)
        
        # Optional: Print stats for the first frame to verify scaling
        print(f"DEBUG [AudioNoise]: Std Dev: {torch.std(s['samples']).item():.4f}")
            
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
