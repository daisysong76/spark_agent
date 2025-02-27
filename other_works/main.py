from music_generator import MusicGenerator

generator = MusicGenerator()

# Generate a 30-second track
audio_path = generator.generate_music(
    prompt="A synthwave track with 80s drums and arpeggiated synths",
    duration=30
)

# Visualize the waveform
generator.visualize_audio(audio_path)