import os
import sys

print("=== Starting Basic System Check ===")
print("Python version:", sys.version)
print("Current working directory:", os.getcwd())

# Check if video exists
video_path = "/Users/daisysong/Desktop/spark_agent/other_works/dolly_forward_simple.mp4"
print("\n=== Checking Video File ===")
print("Checking video path:", video_path)
print("File exists:", os.path.exists(video_path))

# Check output directory
output_dir = "/Users/daisysong/Desktop/spark_agent/other_works/music_video_output"
print("\n=== Checking Output Directory ===")
print("Output directory path:", output_dir)
try:
    os.makedirs(output_dir, exist_ok=True)
    print("Directory created/verified successfully")
    
    # Test write permissions
    test_file = os.path.join(output_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("test")
    os.remove(test_file)
    print("Write permission test passed")
except Exception as e:
    print("Error with output directory:", str(e))

# Check critical imports one by one
print("\n=== Checking Dependencies ===")

dependencies = [
    "torch",
    "numpy",
    "cv2",
    "librosa",
    "soundfile",
    "PIL",
    "moviepy",
    "transformers"
]

for dep in dependencies:
    try:
        __import__(dep)
        print(f"✓ {dep} imported successfully")
    except ImportError as e:
        print(f"✗ {dep} import failed:", str(e))

print("\n=== Basic Check Complete ===")

# Only proceed if all checks pass
try:
    print("\n=== Importing VideoMusicGenerator ===")
    from video_music_generator import VideoMusicGenerator
    print("VideoMusicGenerator imported successfully")
except Exception as e:
    print("Error importing VideoMusicGenerator:", str(e))
    sys.exit(1)

print("\n=== Starting Main Process ===")
try:
    generator = VideoMusicGenerator(output_dir=output_dir)
    print("Generator initialized successfully")
except Exception as e:
    print("Error initializing generator:", str(e))
    import traceback
    traceback.print_exc()

# The output files will be saved to the directory /Users/daisysong/Desktop/spark_agent/other_works/music_video_output, which will be created if it doesn't exist. This directory will contain:

# The generated lyrics file
# The music audio file
# The final music video with lyrics overlay

# If your video is short (which I'm guessing it might be since it was created with the dolly forward effect), the system will automatically adjust the lyrics and music to match the duration.

try:
    from moviepy.editor import VideoFileClip
    print("moviepy.editor imported successfully")
except ImportError as e:
    print("Error importing moviepy.editor:", str(e))