from PIL import Image
import math
import os
from moviepy.editor import ImageSequenceClip
import imageio
import numpy as np
import io
from tqdm import tqdm
import struct
import cv2

"""
####################################################################################################################################################################
"""
# Simple XOR encryption/decryption function
def encrypt_decrypt(data, key):
    return bytearray([b ^ key for b in data])

"""
####################################################################################################################################################################
"""

def image_to_encrypted_binary(image_path, encryption_key=123, output_file='encrypted_image.bin'):
    """
    Converts an RGB image to an encrypted binary file.
    
    Args:
        image_path (str): Path to the input image.
        encryption_key (int): Key used for XOR encryption (default: 123).
        output_file (str): Path to save the encrypted binary file.
    """
    # Open the image and convert it to RGB
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        width, height = img.size
        raw_data = img.tobytes()
    
    # Encrypt the raw RGB data
    encrypted_data = encrypt_decrypt(raw_data, encryption_key)
    
    # Pack width and height into 8 bytes (4 bytes each, big endian)
    header = struct.pack('>II', width, height)
    
    # Save the header followed by the encrypted data
    with open(output_file, 'wb') as f:
        f.write(header)
        f.write(encrypted_data)
    
    print(f"Encrypted data saved as {output_file}")

"""
####################################################################################################################################################################
"""

def encrypted_binary_to_image(encrypted_file, encryption_key=123, output_image='reconstructed_image.png'):
    """
    Decrypts an encrypted binary file and reconstructs the original RGB image.
    
    Args:
        encrypted_file (str): Path to the encrypted binary file.
        encryption_key (int): Key used for XOR decryption (default: 123).
        output_image (str): Path to save the reconstructed image.
    """
    with open(encrypted_file, 'rb') as f:
        # Read the first 8 bytes to get width and height
        header = f.read(8)
        if len(header) < 8:
            raise ValueError("Encrypted file is too short to contain valid header information.")
        
        width, height = struct.unpack('>II', header)
        
        # Read the remaining data as encrypted RGB bytes
        encrypted_data = f.read()
    
    # Decrypt the RGB data
    decrypted_data = encrypt_decrypt(encrypted_data, encryption_key)
    
    # Reconstruct the image from decrypted RGB data
    img = Image.frombytes('RGB', (width, height), decrypted_data)
    img.save(output_image)
    
    print(f"Image recreated and saved as {output_image}")
"""
####################################################################################################################################################################
Encrypted Binary to Video and Video to Encrypted Binary functions
"""

def encrypted_binary_to_video(encrypted_file, output_video='output_video.mp4', width=1920, height=1080, fps=24):
    # Read the encrypted binary file
    with open(encrypted_file, 'rb') as f:
        encrypted_data = f.read()

    # Decrypt the data
    decrypted_data = encrypt_decrypt(encrypted_data, 123)
    
    # Convert decrypted data to binary string
    binary_string = ''.join(format(byte, '08b') for byte in decrypted_data)

    # Calculate how many pixels are needed to represent the binary data
    pixels_per_image = width * height
    num_pixels = len(binary_string)

    # Calculate number of frames required
    num_images = math.ceil(num_pixels / pixels_per_image)
    frames = []

    # Add padding to binary string to ensure it fits perfectly in the last frame
    while len(binary_string) % pixels_per_image != 0:
        binary_string += '0'  # Pad with zeros if necessary

    for i in tqdm(range(num_images)):
        start_index = i * pixels_per_image
        end_index = min(start_index + pixels_per_image, num_pixels)
        binary_digits = binary_string[start_index:end_index]

        img = Image.new('RGB', (width, height), color='white')

        # Convert the binary digits to pixel colors
        for row_index in range(height):
            start_index = row_index * width
            end_index = start_index + width
            row = binary_digits[start_index:end_index]

            for col_index, digit in enumerate(row):
                color = (0, 0, 0) if digit == '1' else (255, 255, 255)
                img.putpixel((col_index, row_index), color)

        frames.append(np.array(img))

    # Create and save the video from frames
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_video, fps=fps)
    print(f"Video created from encrypted binary and saved as {output_video}")

"""
####################################################################################################################################################################
"""

def ExtractFrames(video_path):
    # Extract frames from the video using OpenCV
    video = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()
    return frames

def video_to_encrypted_binary(video_path, output_file='encrypted_video.bin', encryption_key=123):
    # Extract frames from the video
    frames = ExtractFrames(video_path)
    binary_data = ""

    # Process each frame to convert it to binary
    for frame in tqdm(frames, desc="Converting video frames to binary"):
        gray_frame = np.mean(frame, axis=2).astype(np.uint8)  # Convert to grayscale
        binary_digits = ''
        threshold = 128  # Threshold to convert grayscale to binary (0 or 1)

        for y in range(gray_frame.shape[0]):
            for x in range(gray_frame.shape[1]):
                color = gray_frame[y, x]
                binary_digits += '1' if color < threshold else '0'

        binary_data += binary_digits

    # Encrypt the binary data
    encrypted_data = encrypt_decrypt(bytearray(int(binary_data[i:i + 8], 2) for i in range(0, len(binary_data), 8)), encryption_key)

    # Write the encrypted data to file
    with open(output_file, 'wb') as f:
        f.write(encrypted_data)
    print(f"Encrypted binary data saved as {output_file}")

""" 
####################################################################################################################################################################
Text to encrypted binary and encrypted binary to text functions
"""
# New functions for text to encrypted binary and encrypted binary to text
def text_to_encrypted_binary(text_file, output_filename='encrypted_binary_text.bin', encryption_key=123):
    with open(text_file, 'r') as f:
        text = f.read()
    binary_string = ''.join(format(ord(char), '08b') for char in text)

    # Convert binary string to bytes
    byte_array = bytearray(int(binary_string[i:i + 8], 2) for i in range(0, len(binary_string), 8))

    # Encrypt the binary data
    encrypted_data = encrypt_decrypt(byte_array, encryption_key)

    with open(output_filename, 'wb') as f:
        f.write(encrypted_data)
    print(f"Text converted to encrypted binary and saved as {output_filename}")

def encrypted_binary_to_text(encrypted_file, output_filename='decrypted_text.txt', decryption_key=123):
    with open(encrypted_file, 'rb') as f:
        encrypted_data = f.read()

    # Decrypt the data
    decrypted_data = encrypt_decrypt(encrypted_data, decryption_key)

    # Convert bytes back to binary string
    binary_string = ''.join(format(byte, '08b') for byte in decrypted_data)

    # Convert binary string to text while avoiding padding null characters
    text = ''.join(chr(int(binary_string[i:i + 8], 2)) for i in range(0, len(binary_string), 8))

    # Filter out any non-printable characters (like null)
    text = ''.join(c for c in text if c.isprintable() or c.isspace())

    with open(output_filename, 'w') as f:
        f.write(text)
    print(f"Encrypted binary converted to text and saved as {output_filename}")
"""
#################################################################################################################################################################### 
"""

# Main menu
if __name__ == "__main__":
    while True:
        input_Data = input(
            "Choose an option:\n"
            "1. Convert image to encrypted binary\n"
            "2. Reconstruct image from encrypted binary\n"
            "3. Convert encrypted binary to video\n"
            "4. Convert video to encrypted binary\n"
            "5. Convert text to encrypted binary\n"
            "6. Convert encrypted binary to text\n"
            "Press any other key to exit.\n"
        )

        if input_Data == "1":
            input_file = input("Enter the path to the image file: ")
            image_to_encrypted_binary(input_file)

        elif input_Data == "2":
            encrypted_file = input("Enter the path to the encrypted binary file: ")
            output_image = input("Enter the output filename for the reconstructed image (default: 'output_image.png'): ") or 'output_image.png'
            encrypted_binary_to_image(encrypted_file)

        elif input_Data == "3":
            encrypted_file = input("Enter the path to the encrypted binary file: ")
            encrypted_binary_to_video(encrypted_file)

        elif input_Data == "4":
            video_file = input("Enter the path to the video file: ")
            video_to_encrypted_binary(video_file)

        elif input_Data == "5":
            text_file = input("Enter the path to the text file: ")
            text_to_encrypted_binary(text_file)

        elif input_Data == "6":
            encrypted_file = input("Enter the path to the encrypted binary file: ")
            encrypted_binary_to_text(encrypted_file)

        else:
            break
