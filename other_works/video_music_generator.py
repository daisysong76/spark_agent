# This project creates an AI agent that:

# Analyzes a video to understand its content
# Generates song lyrics based on what it sees
# Creates music that matches the video's mood
# Combines everything into a music video with timed lyrics

#pip install torch numpy opencv-python librosa soundfile moviepy transformers pillow

# Prepare your environment:
# You'll need about 8GB of RAM minimum
# A GPU is highly recommended for faster processing
# The first run will download several AI models (5-10GB total)


# Input video requirements:
# Use a standard video format (MP4, MOV, AVI)
# Length: 30 seconds to 3 minutes works best
# The video should have clear visual elements for the AI to detect

# Customization options:
# style: Change music style ("pop", "rock", "electronic", etc.)
# output_dir: Where to save all generated files
# Advanced settings for lyrics generation and music creation inside the class methods


import os
import torch
import numpy as np
import cv2
import librosa
import soundfile as sf
import subprocess
from PIL import Image, ImageDraw, ImageFont
try:
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip
except ImportError:
    print("Trying alternative moviepy import...")
    import moviepy.editor as mpy
    VideoFileClip = mpy.VideoFileClip
    AudioFileClip = mpy.AudioFileClip
    CompositeVideoClip = mpy.CompositeVideoClip
    TextClip = mpy.TextClip
from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM, 
    MusicgenForConditionalGeneration, 
    pipeline,
    AutoImageProcessor, 
    AutoModelForObjectDetection
)

class VideoMusicGenerator:
    def __init__(self, output_dir="output"):
        """Initialize the video music generator with all required models."""
        print("Initializing VideoMusicGenerator...")
        
        # Check if output directory is writable
        try:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
            test_file = os.path.join(output_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            print(f"Output directory {output_dir} is writable")
        except Exception as e:
            print(f"Error: Cannot write to output directory {output_dir}: {str(e)}")
            raise
        
        # Device setup with more detailed info
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
            if self.device == "cuda":
                print(f"GPU: {torch.cuda.get_device_name()}")
                print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                print("Warning: Running on CPU. This will be slow!")
        except Exception as e:
            print(f"Error setting up device: {str(e)}")
            raise
        
        try:
            # Initialize component models with progress indicators
            print("\nInitializing models (this may take a few minutes)...")
            
            print("\n1/3 Loading video analysis model...")
            self._init_video_analysis_model()
            
            print("\n2/3 Loading lyrics generation model...")
            self._init_lyrics_generation_model()
            
            print("\n3/3 Loading music generation model...")
            self._init_music_generation_model()
            
            print("\nAll models initialized successfully!")
        except Exception as e:
            print(f"\nError initializing models: {str(e)}")
            raise
        
    def _init_video_analysis_model(self):
        """Initialize model for video content/scene analysis."""
        try:
            print("Downloading video analysis model (if needed)...")
            self.image_processor = AutoImageProcessor.from_pretrained(
                "facebook/detr-resnet-50",
                local_files_only=False,
                cache_dir="model_cache"
            )
            print("Image processor loaded")
            
            self.detection_model = AutoModelForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50",
                local_files_only=False,
                cache_dir="model_cache"
            ).to(self.device)
            print("Detection model loaded and moved to", self.device)
            
        except Exception as e:
            print(f"Error loading video analysis model: {str(e)}")
            raise
        
    def _init_lyrics_generation_model(self):
        """Initialize model for lyrics generation."""
        try:
            print("Loading lyrics generation model...")
            self.lyrics_generator = pipeline("text-generation", model="openai-community/gpt2-xl", local_files_only=False)
            print("Lyrics generation model loaded successfully!")
        except Exception as e:
            print(f"Error loading lyrics generation model: {str(e)}")
            raise
        
    def _init_music_generation_model(self):
        """Initialize model for music generation."""
        try:
            print("Loading music generation model...")
            model_name = "facebook/musicgen-small"  # Use small model for faster processing
            self.music_processor = AutoProcessor.from_pretrained(model_name, local_files_only=False)
            self.music_model = MusicgenForConditionalGeneration.from_pretrained(model_name, local_files_only=False).to(self.device)
            print("Music generation model loaded successfully!")
        except Exception as e:
            print(f"Error loading music generation model: {str(e)}")
            raise
    
    def analyze_video(self, video_path, sample_rate=1):
        """
        Analyze video content to extract themes, objects, and scenes.
        
        Args:
            video_path: Path to the input video
            sample_rate: How many frames to sample per second
            
        Returns:
            Dictionary with video analysis results
        """
        print(f"Analyzing video: {video_path}")
        
        # Load video
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        # Sample frames at regular intervals
        frame_indices = np.linspace(0, frame_count-1, int(duration * sample_rate), dtype=int)
        
        # Collect objects from sampled frames
        objects_detected = []
        scenes = []
        colors = []
        
        for i in frame_indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            if not ret:
                continue
                
            # Convert BGR to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract dominant colors
            pixels = np.float32(frame_rgb.reshape(-1, 3))
            n_colors = 5
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
            flags = cv2.KMEANS_RANDOM_CENTERS
            _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
            dominant_color = palette[np.argmax(np.bincount(labels))]
            colors.append(dominant_color.tolist())
            
            # Detect objects in the frame
            inputs = self.image_processor(images=Image.fromarray(frame_rgb), return_tensors="pt").to(self.device)
            outputs = self.detection_model(**inputs)
            
            # Keep detections with confidence > 0.7
            confident_detections = []
            for score, label, box in zip(outputs.logits[0].softmax(-1).max(-1).values, 
                                        outputs.logits[0].softmax(-1).argmax(-1), 
                                        outputs.pred_boxes[0]):
                if score > 0.7:
                    label_name = self.detection_model.config.id2label[label.item()]
                    confident_detections.append(label_name)
            
            if confident_detections:
                objects_detected.extend(confident_detections)
            
            # Basic scene analysis
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            avg_brightness = np.mean(hsv[:,:,2])
            if avg_brightness < 80:
                scene_type = "dark"
            elif avg_brightness > 180:
                scene_type = "bright"
            else:
                scene_type = "medium"
            
            scenes.append(scene_type)
        
        video.release()
        
        # Count object frequencies
        object_counts = {}
        for obj in objects_detected:
            object_counts[obj] = object_counts.get(obj, 0) + 1
        
        # Get top objects
        top_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Determine dominant scene type
        scene_counts = {}
        for scene in scenes:
            scene_counts[scene] = scene_counts.get(scene, 0) + 1
        dominant_scene = max(scene_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate average color
        avg_color = np.mean(colors, axis=0).astype(int).tolist()
        
        analysis_results = {
            "duration": duration,
            "fps": fps,
            "top_objects": top_objects,
            "dominant_scene": dominant_scene,
            "avg_color": avg_color,
            "frame_count": frame_count
        }
        
        print(f"Video analysis complete. Duration: {duration:.2f}s, Top objects: {[obj[0] for obj in top_objects[:3]]}")
        return analysis_results
    
    def generate_lyrics(self, video_analysis, style="pop", verses=2, chorus_repeats=2):
        """
        Generate lyrics based on video content analysis.
        
        Args:
            video_analysis: Dictionary with video analysis results
            style: Music style for lyrics
            verses: Number of verses to generate
            chorus_repeats: Number of times to repeat the chorus
            
        Returns:
            Dictionary with generated lyrics and timing information
        """
        print("Generating lyrics based on video analysis...")
        
        # Create prompt from video analysis
        top_objects = [obj[0] for obj in video_analysis["top_objects"][:5]]
        mood = "dark" if video_analysis["dominant_scene"] == "dark" else "uplifting"
        
        prompt = f"Write {style} song lyrics about {', '.join(top_objects)}. The mood is {mood}. Include {verses} verses and a chorus. Format with 'Verse 1:', 'Chorus:', etc."
        
        # Generate lyrics
        lyrics_output = self.lyrics_generator(
            prompt,
            max_length=512,
            num_return_sequences=1,
            temperature=0.8,
            top_p=0.9,
        )[0]["generated_text"]
        
        # Extract just the lyrics part (skip the prompt)
        lyrics_text = lyrics_output[len(prompt):]
        
        # Basic timing calculation
        duration = video_analysis["duration"]
        verse_duration = duration / (verses + chorus_repeats)
        
        # Structure lyrics with timing
        structured_lyrics = self._structure_lyrics(lyrics_text, duration, verses, chorus_repeats)
        
        print(f"Lyrics generated: {len(structured_lyrics['lyrics_text'])} characters")
        return structured_lyrics
    
    def _structure_lyrics(self, lyrics_text, duration, verses, chorus_repeats):
        """Structure the raw lyrics text with timing information."""
        
        # Clean up the text
        lyrics_text = lyrics_text.strip()
        
        # Simple parsing to identify verse and chorus segments
        sections = []
        lines = lyrics_text.split('\n')
        current_section = {"type": "unknown", "lines": []}
        
        for line in lines:
            if line.lower().startswith("verse") or line.lower().startswith("v"):
                if current_section["lines"]:
                    sections.append(current_section)
                current_section = {"type": "verse", "lines": []}
            elif line.lower().startswith("chorus") or line.lower().startswith("c"):
                if current_section["lines"]:
                    sections.append(current_section)
                current_section = {"type": "chorus", "lines": []}
            elif line.strip():
                current_section["lines"].append(line)
                
        if current_section["lines"]:
            sections.append(current_section)
        
        # If parsing failed, create a basic structure
        if not sections or all(s["type"] == "unknown" for s in sections):
            all_lines = [l for l in lines if l.strip()]
            line_count = len(all_lines)
            
            # Divide lines into verses and chorus
            verse_lines = int(line_count * 0.8 / verses)
            chorus_lines = line_count - (verse_lines * verses)
            
            sections = []
            line_index = 0
            
            for i in range(verses):
                verse = {"type": "verse", "lines": []}
                for j in range(verse_lines):
                    if line_index < line_count:
                        verse["lines"].append(all_lines[line_index])
                        line_index += 1
                sections.append(verse)
                
                if i == 0:  # Add chorus after first verse
                    chorus = {"type": "chorus", "lines": []}
                    for j in range(chorus_lines):
                        if line_index < line_count:
                            chorus["lines"].append(all_lines[line_index])
                            line_index += 1
                    sections.append(chorus)
        
        # Calculate timing based on duration and sections
        section_durations = []
        verse_time = duration / (verses + chorus_repeats)
        chorus_time = verse_time
        
        for section in sections:
            if section["type"] == "verse":
                section_durations.append(verse_time)
            elif section["type"] == "chorus":
                section_durations.append(chorus_time)
            else:
                section_durations.append(verse_time / 2)
        
        total_allocated = sum(section_durations)
        scaling_factor = duration / total_allocated if total_allocated > 0 else 1
        
        # Apply scaling factor to match video duration
        section_durations = [d * scaling_factor for d in section_durations]
        
        # Create timed lyrics
        current_time = 0
        timed_lyrics = []
        
        for i, (section, duration) in enumerate(zip(sections, section_durations)):
            section_lines = section["lines"]
            if not section_lines:
                continue
                
            line_duration = duration / len(section_lines)
            
            for line in section_lines:
                timed_lyrics.append({
                    "text": line,
                    "start_time": current_time,
                    "end_time": current_time + line_duration,
                    "section_type": section["type"]
                })
                current_time += line_duration
        
        # Compile full lyrics text
        full_lyrics = "\n\n".join([
            f"{section['type'].title()}:\n" + "\n".join(section['lines'])
            for section in sections
            if section['lines']
        ])
        
        return {
            "lyrics_text": full_lyrics,
            "timed_lyrics": timed_lyrics,
            "sections": sections
        }
    
    def generate_music(self, video_analysis, lyrics):
        """
        Generate music based on video content and lyrics.
        
        Args:
            video_analysis: Dictionary with video analysis results
            lyrics: Dictionary with generated lyrics information
            
        Returns:
            Path to generated music file
        """
        print("Generating music...")
        
        # Determine music style based on video content
        top_objects = [obj[0] for obj in video_analysis["top_objects"][:3]]
        mood = "dark" if video_analysis["dominant_scene"] == "dark" else "upbeat"
        
        # Create descriptive prompt for music generation
        music_prompt = f"Create a {mood} track inspired by scenes with {', '.join(top_objects)}. The song should match lyrics about these themes."
        
        # Calculate music duration (slightly longer than video)
        duration = video_analysis["duration"] + 2  # Add 2 seconds buffer
        
        # Convert to required tokens for generation
        max_new_tokens = int(duration * self.music_model.config.audio_encoder.sampling_rate / self.music_model.config.audio_encoder.hop_length)
        
        # Prepare input
        inputs = self.music_processor(
            text=[music_prompt],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Generate music
        print(f"Generating approximately {duration:.1f} seconds of music...")
        audio_values = self.music_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            guidance_scale=3.0,
            temperature=1.0,
        )
        
        # Convert to numpy array and save
        sampling_rate = self.music_model.config.audio_encoder.sampling_rate
        audio_numpy = audio_values[0, 0].cpu().numpy()
        
        # Save the audio file
        music_path = os.path.join(self.output_dir, "generated_music.wav")
        sf.write(music_path, audio_numpy, sampling_rate)
        
        print(f"Music generated and saved to: {music_path}")
        return music_path
    
    def create_lyric_video(self, video_path, music_path, lyrics, output_path=None):
        """
        Create a music video with lyrics overlay.
        
        Args:
            video_path: Path to the original video
            music_path: Path to the generated music
            lyrics: Dictionary with lyrics and timing information
            output_path: Path for the output video file
            
        Returns:
            Path to the final music video
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "final_music_video.mp4")
            
        print(f"Creating lyric video: {output_path}")
        
        # Load video and audio
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(music_path)
        
        # Resize audio if needed
        if audio_clip.duration > video_clip.duration:
            audio_clip = audio_clip.subclip(0, video_clip.duration)
        elif audio_clip.duration < video_clip.duration:
            # Extend by looping if too short
            times_to_loop = int(np.ceil(video_clip.duration / audio_clip.duration))
            audio_clip = audio_clip.loop(times_to_loop)
            audio_clip = audio_clip.subclip(0, video_clip.duration)
        
        # Set audio to video
        video_with_audio = video_clip.set_audio(audio_clip)
        
        # Create lyric overlays
        lyric_clips = []
        
        for lyric in lyrics["timed_lyrics"]:
            text = lyric["text"]
            start_time = lyric["start_time"]
            end_time = lyric["end_time"]
            
            # Create text clip with styling based on section type
            font_size = 40 if lyric["section_type"] == "chorus" else 30
            font_color = 'yellow' if lyric["section_type"] == "chorus" else 'white'
            
            text_clip = TextClip(
                text, 
                fontsize=font_size, 
                color=font_color,
                font='Arial-Bold',
                stroke_color='black',
                stroke_width=2,
                method='caption',
                size=(video_clip.w * 0.9, None),
                align='center'
            )
            
            # Position at bottom of screen
            text_clip = text_clip.set_position(('center', 'bottom')).margin(bottom=50, opacity=0)
            
            # Set timing
            text_clip = text_clip.set_start(start_time).set_end(end_time)
            
            # Add to list
            lyric_clips.append(text_clip)
        
        # Combine video with lyrics
        final_clip = CompositeVideoClip([video_with_audio] + lyric_clips)
        
        # Write final video
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            fps=video_clip.fps
        )
        
        # Clean up
        video_clip.close()
        audio_clip.close()
        final_clip.close()
        
        print(f"Music video created: {output_path}")
        return output_path
    
    def process_video(self, video_path, style="pop", output_path=None):
        """
        Process a video to create a complete music video with generated lyrics and music.
        
        Args:
            video_path: Path to the input video
            style: Style of music to generate
            output_path: Path for the output video file
            
        Returns:
            Dictionary with paths to all generated files
        """
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(self.output_dir, f"{base_name}_music_video.mp4")
        
        print(f"Starting complete processing of video: {video_path}")
        
        # Step 1: Analyze video
        video_analysis = self.analyze_video(video_path)
        
        # Step 2: Generate lyrics
        lyrics = self.generate_lyrics(video_analysis, style=style)
        
        # Save lyrics to file
        lyrics_path = os.path.join(self.output_dir, "generated_lyrics.txt")
        with open(lyrics_path, "w") as f:
            f.write(lyrics["lyrics_text"])
        
        # Step 3: Generate music
        music_path = self.generate_music(video_analysis, lyrics)
        
        # Step 4: Create lyric video
        final_video_path = self.create_lyric_video(video_path, music_path, lyrics, output_path)
        
        result = {
            "input_video": video_path,
            "lyrics_file": lyrics_path,
            "music_file": music_path,
            "output_video": final_video_path
        }
        
        print(f"Complete processing finished. Output video: {final_video_path}")
        return result


# Example usage
if __name__ == "__main__":
    # Replace with your video file
    input_video = "input_video.mp4"
    
    # Initialize the generator
    generator = VideoMusicGenerator(output_dir="output")
    
    # Process the video
    result = generator.process_video(
        video_path=input_video,
        style="pop"  # Try: "rock", "hip-hop", "electronic", "country", etc.
    )
    
    print("\nProcessing complete! Generated files:")
    for key, path in result.items():
        print(f"- {key}: {path}")