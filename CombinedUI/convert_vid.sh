# Define the output folder
output_folder="."

# Create the output folder if it doesn't exist
mkdir -p "$output_folder"

# Process each .mp4 file in the current folder
for file in *.mp4; do
    # Replace spaces with underscores in the filename
    new_name=$(echo "$file" | tr ' ' '_')

    # Rename the file to remove spaces
    mv "$file" "$new_name"

    # Extract the base name without the extension
    base_name="${new_name%.*}"

    # Re-encode the video and place it in the output folder
    ffmpeg -i "$new_name" -c:v mpeg4 -q:v 2 -c:a aac "$output_folder/${base_name}.mp4"
done
