import subprocess
import os

def process_video(input_path, output_format="mp4"):
    # Check if the file exists
    if not os.path.isfile(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        return

    # Get file name and output path
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = f"{base_name}_processed.{output_format}"

    # FFmpeg command to convert the video
    command = [
        "ffmpeg", "-i", input_path,  # Input file
        "-c:v", "libx264",          # Video codec
        "-preset", "fast",          # Speed preset
        "-c:a", "aac",              # Audio codec
        "-b:a", "128k",             # Audio bitrate
        output_path                 # Output file
    ]

    try:
        print("Processing video... This may take a while.")
        subprocess.run(command, check=True)
        print(f"Video processing complete! File saved as '{output_path}'")
    except subprocess.CalledProcessError as e:
        print("Error occurred while processing the video:", e)

if __name__ == "__main__":
    # Replace 'input_file_path' with the path to your video file
    input_file_path = "your_video_file_here.mkv"  # Update this to your video file path
    process_video(input_file_path)
