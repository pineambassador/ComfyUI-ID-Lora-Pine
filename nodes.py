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
                "dropout": ("CONDITIONING",),
                "audio": ("AUDIO",),
                "audio_vae": ("VAE",),
                "audio_strength": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 10.0, "step": 0.1}),
                "timbre_boost": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 4.0, "step": 0.05}),
                "pitch_adjustment": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
                "audio_distance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("conditioning", "dropout",)
    FUNCTION = "process_dual_audio"
    CATEGORY = "ID-LoRA/Audio"

    def process_dual_audio(self, conditioning, dropout, audio, audio_vae, audio_strength, timbre_boost, pitch_adjustment, audio_distance):
        TARGET_RATE = 24000 
        
        try:
            waveform = audio["waveform"]
            sr = audio["sample_rate"]

            # Convert to Mono if necessary
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # --- 1. PITCH SHIFT LOGIC ---
            effective_sr = int(sr / pitch_adjustment)
            import torchaudio.transforms as T
            resampler = T.Resample(effective_sr, TARGET_RATE).to(waveform.device)
            waveform = resampler(waveform.to(torch.float32))

            # --- 2. VAE ENCODING ---
            vae_input = {"waveform": waveform, "sample_rate": TARGET_RATE}
            latents_dict = audio_vae.encode(vae_input)
            audio_samples = latents_dict.get("samples") if isinstance(latents_dict, dict) else latents_dict

            # --- 3. SYMMETRY & NORMALIZATION ---
            raw_std = torch.std(audio_samples).item()
            raw_mean = torch.mean(audio_samples)
            
            # Center the signal to 0
            audio_samples = audio_samples - raw_mean
            
            # Force unit variance (Unity Gain)
            if raw_std > 0:
                audio_samples = audio_samples / raw_std

            # --- 4. TIMBRE & DISTANCE PERSPECTIVE ---
            # Calculate distance roll-off for timbre
            distance_factor = max(0.1, 1.0 - (audio_distance * 0.2))
            current_timbre = max(1.0, timbre_boost * distance_factor)
            audio_samples = audio_samples * current_timbre
            
            # Soft Clip to prevent 'Robotic' stuttering (The -4 to 4 safety zone)
            audio_samples = torch.tanh(audio_samples / 4.0) * 4.0
            
            # Apply volume drop for distance
            final_strength = audio_strength / (1.0 + audio_distance)
            audio_samples = audio_samples * final_strength

            # Prepare for injection (unsqueeze to 5D for LTX conditioning)
            while audio_samples.dim() < 5:
                audio_samples = audio_samples.unsqueeze(-1)

            # DIAGNOSTICS
            print(f"--- [DUAL-PATH AUDIO DIAGNOSTICS] ---")
            print(f"Pitch: {pitch_adjustment}x | Timbre: {current_timbre:.2f}")
            print(f"Final Std Dev: {torch.std(audio_samples).item():.4f}")
            print(f"Latent Range: [{torch.min(audio_samples).item():.2f} to {torch.max(audio_samples).item():.2f}]")

            # --- 5. INJECTION LOGIC (SYMMETRIC) ---
            def apply_audio_to_cond(cond_list):
                new_conditioning = []
                for t in cond_list:
                    metadata = t[1].copy()
                    # Staple the processed audio to the metadata
                    metadata["audio_latent"] = audio_samples.clone()
                    # Ensure weight is consistent
                    metadata["audio_weight"] = 1.0 
                    new_conditioning.append([t[0].clone(), metadata])
                return new_conditioning

            # Inject the same exact latents into both branches
            pos_out = apply_audio_to_cond(conditioning)
            drop_out = apply_audio_to_cond(dropout)

            return (pos_out, drop_out)

        except Exception as e:
            print(f"ERROR in IDLoRAPrepareAudioReference: {e}")
            return (conditioning, dropout)


# --- VIDEO PREPARATION (First Frame Injection) ---

class IDLoRAPrepareVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_video": ("LATENT",),
                "first_frame_ref": ("LATENT",),
                "strength": ("FLOAT", {"default": 0.92, "min": 0.0, "max": 1.0, "step": 0.01}),
                "portrait_strength": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
                "frame_positions": ("STRING", {"default": "1, 48"}),
                "fade_frames": ("INT", {"default": 10, "min": 1, "max": 50}),
                "mode": (["Base", "Refiner"], {"default": "Base"}),
            },
            "optional": {
                "portraits": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "prepare"
    CATEGORY = "ID-LoRA/Video"

    def prepare(self, latent_video, first_frame_ref, strength, portrait_strength, frame_positions, fade_frames, mode, portraits=None):
        v = latent_video["samples"].clone()
        ref = first_frame_ref["samples"].clone()
        B, C, F, H, W = v.shape

        # 1. SCALING CORRECTION (Reference)
        if torch.std(ref).item() < 0.5: ref = ref / 0.18215

        mask = torch.ones((B, 1, F, H, W), device=v.device, dtype=v.dtype)
        
        # 2. DETECT INPUT TYPE (Image vs. Video Seed)
        f_ref_count = ref.shape[2] if ref.dim() == 5 else 1

        if mode == "Base":
            # --- VIDEO/IMAGE ANCHORING ---
            for i in range(f_ref_count):
                if i < F:
                    current_f = ref[:, :, i, :, :] if ref.dim() == 5 else ref
                    if current_f.dim() == 4 and current_f.shape[0] > B:
                        current_f = current_f[0:B]
                    
                    if current_f.shape[-2:] != (H, W):
                        current_f = torch.nn.functional.interpolate(current_f, size=(H, W), mode="bilinear")
                    
                    v[:, :, i, :, :] = (current_f * strength) + (v[:, :, i, :, :] * (1.0 - strength))
                    mask[:, :, i, :, :] = 1.0 - strength 

            # --- OPTIONAL PORTRAITS WITH FADE ---
            if portraits is not None:
                pts = portraits["samples"].clone()
                if torch.std(pts).item() < 0.5: pts = pts / 0.18215
                
                try:
                    pos_list = [int(x.strip()) for x in frame_positions.split(",")]
                except:
                    pos_list = [1, 48]

                num_pts = pts.shape[0]
                for i in range(num_pts):
                    target_f = pos_list[i] if i < len(pos_list) else pos_list[-1] + (i * 10)
                    
                    p_img = pts[i:i+1].squeeze(2) if pts.dim() == 5 else pts[i:i+1]
                    if p_img.shape[-2:] != (H, W):
                        p_img = torch.nn.functional.interpolate(p_img, size=(H, W), mode="bilinear")

                    for f_offset in range(fade_frames):
                        current_idx = target_f + f_offset
                        if current_idx >= F: break
                        
                        fade_factor = 1.0 - (f_offset / fade_frames)
                        current_p_strength = portrait_strength * fade_factor
                        
                        v[:, :, current_idx, :, :] = (p_img * current_p_strength) + (v[:, :, current_idx, :, :] * (1.0 - current_p_strength))

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
                "cfg": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1, "label": "Global CFG"}),
                "id_cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1, "label": "Identity CFG"}),
                "audio_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1, "label": "Audio CFG"}),
            },
            "optional": {
                "id_dropout_cond": ("CONDITIONING",), 
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
            m["id_weight"] = id_cfg      
            m["audio_weight"] = audio_cfg 
            
            # The Sampler uses this text-only latent to calculate the 'Delta'
            if id_dropout_cond is not None:
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
                "secondary_desc": ("STRING", {"multiline": True, "default": ""}), 
                "visual_action": ("STRING", {"multiline": True, "default": "nodding slowly while looking at the camera"}),
                "dialogue_text": ("STRING", {"multiline": True, "default": "I think we've finally found the solution."}),
                "environmental_sounds": ("STRING", {"multiline": True, "default": "soft rain"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "low quality, blurry..."}),
            }
        }
    
    # We now return three strings instead of two
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "dropout_prompt", "negative_prompt")
    FUNCTION = "format_prompt"
    CATEGORY = "ID-LoRA/Prompting"

    def format_prompt(self, id_tag, primary_desc, secondary_desc, visual_action, dialogue_text, environmental_sounds, negative_prompt):
        # 1. FULL CONDITIONED PROMPT (For the main 'conditioning' input)
        if secondary_desc.strip() and secondary_desc.lower() != "none":
            visual_block = (
                f" [VISUAL]: {id_tag} {primary_desc}, "
                f"and {secondary_desc}. "
                f"{visual_action}."
            )
        else:
            visual_block = f" [VISUAL]: {id_tag} {primary_desc}, {visual_action}\n"

        speech_block = f"[SPEECH]: {dialogue_text}\n"
        sound_block = f"[SOUNDS]: {environmental_sounds}\n"
        
        full_pos = f"{visual_block} {speech_block} {sound_block}"

        # 2. DROPOUT PROMPT (The "Clean" version for Delta calculation)
        # We remove the [NAME] tag and the specific [SPEECH]/[SOUNDS] markers
        # This gives the model a 'generic' baseline to compare against.
        if secondary_desc.strip() and secondary_desc.lower() != "none":
            dropout_pos = f"two people, {environmental_sounds}."
        else:
            dropout_pos = f"a person, {environmental_sounds}."

        return (full_pos, dropout_pos, negative_prompt)

class IDLoRAAudioNoiseInjector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "apply_scaling": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "inject_noise"
    CATEGORY = "ID-LoRA/Audio"

    def inject_noise(self, samples, seed, strength, apply_scaling):
        s = samples.copy()
        audio_tensor = s["samples"].clone()
        
        if apply_scaling:
            audio_tensor = audio_tensor * 0.18215
        
        torch.manual_seed(seed)
        noise = torch.randn_like(audio_tensor)
        s["samples"] = (audio_tensor * (1.0 - strength)) + (noise * strength)
            
        return (s,)


NODE_CLASS_MAPPINGS = {
    "IDLoRAGuider": IDLoRAGuider,
    "IDLoRAPrepareVideo": IDLoRAPrepareVideo,
    "IDLoRAPromptFormatter": IDLoRAPromptFormatter,
    "IDLoRAAudioNoiseInjector": IDLoRAAudioNoiseInjector,
    "IDLoRAPrepareAudioReference": IDLoRAPrepareAudioReference,
}

NODE_DISPLAY_NAME_MAPPINGS = {}
