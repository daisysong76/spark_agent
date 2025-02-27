# pip install opencv-python numpy pillow torch diffusers matplotlib

import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import StableVideoDiffusionPipeline
import matplotlib.pyplot as plt

def create_dolly_forward_video(image_path, output_path, num_frames=24, zoom_factor=1.5):
    """
    Simple implementation using OpenCV for a dolly forward effect
    
    Args:
        image_path: Path to the input image
        output_path: Path for the output video
        num_frames: Number of frames in the output video
        zoom_factor: How much to zoom in by the end of the video
    """
    # Load the image
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, 24, (width, height))
    
    # Create frames with progressive zoom (dolly forward effect)
    for i in range(num_frames):
        # Calculate zoom for this frame
        current_zoom = 1 + (zoom_factor - 1) * i / (num_frames - 1)
        
        # Calculate crop dimensions to simulate dolly forward
        new_width = int(width / current_zoom)
        new_height = int(height / current_zoom)
        
        # Calculate crop position (center crop)
        x = (width - new_width) // 2
        y = (height - new_height) // 2
        
        # Crop image and resize back to original dimensions
        cropped = img[y:y+new_height, x:x+new_width]
        frame = cv2.resize(cropped, (width, height))
        
        # Add frame to video
        video.write(frame)
    
    # Release the video
    video.release()
    print(f"Video saved to {output_path}")

def advanced_video_generation(image_path, output_path):
    """
    More advanced implementation using Stable Video Diffusion
    
    Args:
        image_path: Path to the input image
        output_path: Path for the output video
    """
    # Load the Stable Video Diffusion model
    model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe = pipe.to("cuda")
    
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((1024, 576))  # SVD requires specific dimensions
    
    # Generate video with dolly forward motion
    generator = torch.manual_seed(42)
    frames = pipe(
        image, 
        decode_chunk_size=8,
        generator=generator,
        motion_bucket_id=180,  # Higher values = more motion
        noise_aug_strength=0.1,
        num_frames=24
    ).frames[0]
    
    # Save frames as video using OpenCV
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, 8, (width, height))
    
    for frame in frames:
        video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    
    video.release()
    print(f"Video saved to {output_path}")

# Example usage:
if __name__ == "__main__":
    # Simple OpenCV-based dolly effect
    create_dolly_forward_video("input_image.jpg", "dolly_forward_simple.mp4")
    
    # Advanced AI-based video generation (requires GPU)
    # advanced_video_generation("input_image.jpg", "dolly_forward_ai.mp4")


#     Framework Options

# Use existing image-to-video models:

# Stable Video Diffusion
# AnimateDiff
# Gen-2 by Runway
# FILM (Frame Interpolation for Large Motion)


# Build a custom solution using:

# Deep learning frameworks (PyTorch/TensorFlow)
# Computer vision libraries (OpenCV)
# 3D reconstruction techniques


# Step-by-Step Implementation Guide

# Prepare your environment:

# Install Python and necessary libraries (OpenCV, PyTorch, diffusers)
# Ensure you have a GPU if using advanced AI models


# Select your approach:

# For a quick prototype: Use the OpenCV method in the code above
# For higher quality results: Use the Stable Video Diffusion approach


# Process the image:

# The simple approach creates a dolly effect by progressively cropping and resizing
# The advanced approach uses AI to generate in-between frames with natural motion


# Enhance the results:

# Add depth estimation to improve realism
# Apply motion blur for smoother transitions
# Fine-tune parameters based on your specific image


# Export the final video:

# Use appropriate codecs (H.264 is widely compatible)
# Adjust frame rates for desired speed of dolly motion



# The simple approach will work on most hardware, while the AI-based approach requires a GPU but produces more natural-looking results.
# Would you like me

# By default, this will run only the create_dolly_forward_video() function since the advanced function is commented out in the main block.
# Running the Advanced Version
# To use the AI-based version with Stable Video Diffusion:

# Uncomment the line # advanced_video_generation("input_image.jpg", "dolly_forward_ai.mp4") in the main block
# You'll need a CUDA-capable GPU with at least 8GB VRAM
# The first run will download the model weights (~5GB)

# Viewing Results
# After running the code:

# Look for the output file (dolly_forward_simple.mp4 or dolly_forward_ai.mp4) in your working directory
# Open it with any video player (VLC, Windows Media Player, QuickTime, etc.)
# You should see a video that zooms in on your image, creating the dolly forward effect

# Customization Options
# You can modify these parameters:

# num_frames: Change video length (default: 24 frames)
# zoom_factor: Adjust how much the camera "moves forward" (default: 1.5x)

# If you don't have a GPU for the advanced version, stick with the simple OpenCV approach which works well for many images and runs on any computer.
# Would you like me to explain any specific part of the implementation in more detail?