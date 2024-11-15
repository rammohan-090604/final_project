1. **Install Required Packages:**
    Install the necessary dependencies using `pip`:
    ```bash
    pip install moviepy imageio numpy pytube tqdm pillow
    ```

2. **List of Required Libraries:**
    The following libraries are used in this project:
    - `moviepy` - for handling video files.
    - `imageio` - for reading video frames.
    - `numpy` - for numerical operations.
    - `pytube` - for downloading YouTube videos.
    - `tqdm` - for displaying progress bars.
    - `Pillow` - for image processing.

## Running the Script
1. To start the script, run the `main.py` file:
    ```bash
    python main.py
    ```

2. You will be prompted with a menu to select one of the following options:

    ```
    1. Convert file to binary and save
    2. Convert video to binary
    3. Convert image to encrypted binary
    4. Reconstruct image from encrypted binary
    5. Convert text to binary
    6. Convert binary to text
    7. Download video from YouTube
    8. Convert encrypted binary to video
    9. Convert video to encrypted binary
    Press any other key to exit.
    ```

## Providing Inputs for Each Option
### 1. **Convert File to Binary and Save**
- **Input Required**: Provide the file path (e.g., `path/to/your/file.txt`).
- **Output**: A video will be generated and saved as `output_video.mp4`.

### 2. **Convert Video to Binary**
- **Input Required**: Provide the video file path (e.g., `path/to/your/video.mp4`).
- **Output**: A binary file will be saved as `reverse.mkv`.

### 3. **Convert Image to Encrypted Binary**
- **Input Required**: Provide the image file path (e.g., `path/to/your/image.png`).
- **Output**: An encrypted binary file will be saved (default: `encrypted_image.bin`).

### 4. **Reconstruct Image from Encrypted Binary**
- **Input Required**: Provide the encrypted binary file path.
- **Output**: An image file will be reconstructed (default: `output_image.png`).

### 5. **Convert Text to Binary**
- **Input Required**: Provide the text file path.
- **Output**: A binary representation of the text will be saved.

### 6. **Convert Binary to Text**
- **Input Required**: Provide the binary file path.
- **Output**: The binary content will be converted back to readable text.

### 7. **Download Video from YouTube**
- **Input Required**: Provide the YouTube URL.
- **Output**: A 1080p video will be downloaded to your current directory.

### 8. **Convert Encrypted Binary to Video**
- **Input Required**: Provide the encrypted binary file path.
- **Output**: A video file will be generated (default: `output_video.mp4`).

### 9. **Convert Video to Encrypted Binary**
- **Input Required**: Provide the video file path.
- **Output**: An encrypted binary file will be saved (default: `encrypted_video.bin`).

## Additional Notes
- Ensure that the input files (e.g., images, text files, video files) exist in the specified paths.
- For encryption and decryption, the default key is set to `123`. Modify it as needed in the functions.
#   f i n a l _ p r o j e c t  
 