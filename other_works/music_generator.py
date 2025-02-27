#pip install torch torchaudio transformers soundfile matplotlib numpy ipython
# Hardware requirements:
# For the small model: 4GB+ VRAM or 8GB+ RAM
# For the medium/large models: 8GB+ VRAM recommended
# This will generate four 15-second music samples with different styles.


import os
import torch
import torchaudio
import numpy as np
import soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from IPython.display import Audio, display
import matplotlib.pyplot as plt

class MusicGenerator:
    def __init__(self, model_name="facebook/musicgen-small", device=None):
        """Initialize the music generator with a specified model.
        
        Args:
            model_name: The name of the model to use ("facebook/musicgen-small", 
                       "facebook/musicgen-medium", or "facebook/musicgen-large")
            device: The device to run the model on (None for auto-detection)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        print(f"Loading model: {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        print("Model loaded successfully!")
        
        # Create output directory if it doesn't exist
        self.output_dir = "generated_music"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_music(self, prompt, duration=10, output_filename=None, seed=None):
        """Generate music based on a text prompt.
        
        Args:
            prompt: Text description of the music to generate
            duration: Duration of the generated audio in seconds
            output_filename: Name for the output file (without extension)
            seed: Random seed for reproducibility (optional)
            
        Returns:
            Path to the generated audio file
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Set the default filename if none provided
        if output_filename is None:
            # Clean prompt to create a filename
            clean_prompt = "".join(c if c.isalnum() else "_" for c in prompt)
            clean_prompt = clean_prompt[:30]  # Limit length
            output_filename = f"{clean_prompt}_{duration}s"
        
        # Generate the music
        print(f"Generating music for prompt: '{prompt}'")
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Generate with max duration set in seconds (converted to steps)
        max_new_tokens = int(duration * self.model.config.audio_encoder.sampling_rate / self.model.config.audio_encoder.hop_length)
        
        audio_values = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            guidance_scale=3.0,
            temperature=1.0,
        )
        
        # Convert to numpy array and save
        sampling_rate = self.model.config.audio_encoder.sampling_rate
        audio_numpy = audio_values[0, 0].cpu().numpy()
        
        # Save the audio file
        output_path = os.path.join(self.output_dir, f"{output_filename}.wav")
        sf.write(output_path, audio_numpy, sampling_rate)
        
        print(f"Generated music saved to: {output_path}")
        return output_path
    
    def visualize_audio(self, audio_path):
        """Create a waveform visualization of the generated audio.
        
        Args:
            audio_path: Path to the audio file
        """
        # Load the audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Plot the waveform
        plt.figure(figsize=(10, 4))
        plt.plot(waveform[0].numpy())
        plt.title("Waveform")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        
        # Save the visualization
        viz_path = audio_path.replace(".wav", "_waveform.png")
        plt.savefig(viz_path)
        plt.close()
        print(f"Waveform visualization saved to: {viz_path}")
        
        return viz_path
    
    def play_audio(self, audio_path):
        """Play the generated audio in a notebook environment.
        
        Args:
            audio_path: Path to the audio file
        """
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            display(Audio(waveform.numpy()[0], rate=sample_rate))
            print(f"Playing audio from: {audio_path}")
        except Exception as e:
            print(f"Could not play audio in this environment: {e}")
            print(f"Audio file is available at: {audio_path}")


def main():
    # Initialize the generator with a small model (faster)
    # Use "facebook/musicgen-medium" or "facebook/musicgen-large" for higher quality
    generator = MusicGenerator("facebook/musicgen-small")
    
    # Example prompts to try
    example_prompts = [
        "An upbeat electronic dance track with a catchy melody and driving beats",
        "A peaceful ambient soundscape with gentle piano and nature sounds",
        "A rock song with electric guitar solos and energetic drums",
        "A jazz piece with smooth saxophone and gentle percussion"
    ]
    
    # Generate music for each prompt
    for i, prompt in enumerate(example_prompts):
        # Generate a 15-second sample
        audio_path = generator.generate_music(
            prompt=prompt,
            duration=15,
            output_filename=f"sample_{i+1}",
            seed=42+i  # Different seed for each generation
        )
        
        # Create visualization
        generator.visualize_audio(audio_path)
        
        # Try to play it (works in notebook environments)
        generator.play_audio(audio_path)
        
        print("\n" + "-"*50 + "\n")


if __name__ == "__main__":
    main()