# import numpy as np
# import os
# import base64
# import cv2
# from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
# from cryptography.hazmat.primitives import padding
# from cryptography.hazmat.backends import default_backend

# def generate_key():
#     """Generate a random AES key."""
#     return os.urandom(32)  # For AES-256, use a 32-byte key

# def encrypt_binary(binary_data, key):
#     """Encrypt binary data using AES."""
#     iv = os.urandom(16)
#     cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
#     encryptor = cipher.encryptor()
#     padder = padding.PKCS7(algorithms.AES.block_size).padder()
#     padded_data = padder.update(binary_data) + padder.finalize()
#     encrypted = encryptor.update(padded_data) + encryptor.finalize()
#     return base64.b64encode(iv + encrypted).decode('utf-8')

# def decrypt_binary(enc_data, key):
#     """Decrypt the encrypted binary data using AES."""
#     enc_data_bytes = base64.b64decode(enc_data)
    
#     # Ensure there are enough bytes to extract the IV
#     if len(enc_data_bytes) < 16:
#         raise ValueError("The encrypted data is too short to contain a valid IV.")
    
#     iv = enc_data_bytes[:16]  # First 16 bytes are the IV
#     encrypted = enc_data_bytes[16:]
    
#     cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
#     decryptor = cipher.decryptor()
#     decrypted_padded = decryptor.update(encrypted) + decryptor.finalize()
    
#     unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
#     decrypted = unpadder.update(decrypted_padded) + unpadder.finalize()
#     return decrypted

# def encrypt_file(file_path, key):
#     """Encrypt the contents of a file."""
#     with open(file_path, 'rb') as f:
#         binary_data = f.read()
#     return encrypt_binary(binary_data, key)

# def decrypt_file(enc_data, key):
#     """Decrypt the encrypted data and return it."""
#     return decrypt_binary(enc_data, key)

# def binary_to_images(binary_data, img_width=720, img_height=1080):
#     """Convert binary data into a list of images with padding if needed."""
#     images = []
#     arr = np.frombuffer(binary_data, dtype=np.uint8)

#     # Check if we have enough data for at least one image
#     required_size = img_width * img_height
#     if len(arr) < required_size:
#         # Calculate how many more bytes are needed
#         padding_size = required_size - len(arr)
#         # Pad with zeros (0x00)
#         pad_data = bytearray([0x00] * padding_size)
#         arr = np.concatenate((arr, np.frombuffer(pad_data, dtype=np.uint8)))

#     # Append the original data length as the first few bytes
#     original_length = len(binary_data)
#     original_length_bytes = original_length.to_bytes(4, byteorder='big')  # 4 bytes for length
#     arr = np.concatenate((np.frombuffer(original_length_bytes, dtype=np.uint8), arr))

#     for i in range(0, len(arr), img_width * img_height):
#         img_data = arr[i:i + img_width * img_height]
#         if len(img_data) < img_width * img_height:
#             break
#         img_data = img_data.reshape((img_height, img_width))
#         images.append(img_data)
    
#     return images

# def create_video_from_images(images, video_path='output_video.avi', fps=10):
#     """Create a video from a list of images."""
#     if not images:
#         print("No images to create a video.")
#         return

#     height, width = images[0].shape
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height), False)
#     for img in images:
#         video_writer.write(img)
#     video_writer.release()
#     print(f"Video saved as: {video_path}")

# def extract_images_from_video(video_path):
#     """Extract images from the video and return as binary data."""
#     cap = cv2.VideoCapture(video_path)
#     binary_data = bytearray()

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         binary_data.extend(frame.flatten())
    
#     cap.release()
#     return bytes(binary_data)

# def images_to_binary(images):
#     """Convert a list of images back into binary data."""
#     binary_data = bytearray()
#     for img in images:
#         binary_data.extend(img.flatten())
#     return bytes(binary_data)

# def retrieve_original_length(binary_data):
#     """Retrieve the original data length from the first 4 bytes."""
#     if len(binary_data) < 4:
#         return 0
#     return int.from_bytes(binary_data[:4], byteorder='big')

# # Example usage
# if __name__ == "__main__":
#     key = generate_key()

#     while True:
#         print("\nChoose an option:")
#         print("1. Encrypt a file and convert it to video")
#         print("2. Extract data from video and decrypt it")
#         print("3. Exit")

#         choice = input("Enter your choice (1/2/3): ")

#         if choice == '1':
#             input_file_path = input("Enter the path of the file to encrypt: ")
#             encrypted_data = encrypt_file(input_file_path, key)
#             images = binary_to_images(base64.b64decode(encrypted_data))
#             if not images:
#                 print("No images generated from the binary data.")
#                 continue
#             video_path = input("Enter the path to save the video file: ")
#             create_video_from_images(images, video_path)
        
#         elif choice == '2':
#             video_path = input("Enter the path of the video file: ")
#             binary_data = extract_images_from_video(video_path)
#             original_length = retrieve_original_length(binary_data)
#             extracted_binary_data = binary_data[4:4 + original_length]  # Skip the length header
#             decrypted_data = decrypt_file(base64.b64encode(extracted_binary_data).decode('utf-8'), key)

#             # Set a default file name if none provided
#             output_file_path = input("Enter the path to save the decrypted file (default: decrypted_output.bin): ")
#             if not output_file_path:
#                 output_file_path = "decrypted_output.bin"  # Default filename

#             with open(output_file_path, 'wb') as f:
#                 f.write(decrypted_data)
#             print(f"Decrypted data has been saved to: {output_file_path}")

#         elif choice == '3':
#             print("Exiting the program.")
#             break

#         else:
#             print("Invalid choice. Please select 1, 2, or 3.")
#  -------------------------------------------------------------------------------------

# from PIL import Image
# import math
# import os
# from moviepy.editor import ImageSequenceClip
# import imageio
# import numpy as np
# import io
# from tqdm import tqdm
# from pytube import YouTube

# # Simple XOR encryption/decryption function
# def encrypt_decrypt(data, key):
#     return bytearray([b ^ key for b in data])

# def file_to_binary(filepath):
#     file_size = os.path.getsize(filepath)
#     binary_string = ""
#     with open(filepath, "rb") as f:
#         for chunk in tqdm(iterable=iter(lambda: f.read(1024), b""), total=math.ceil(file_size / 1024), unit="KB"):
#             binary_string += "".join(f"{byte:08b}" for byte in chunk)
#     return binary_string

# def binary_to_video(bin_string, width=1920, height=1080, pixel_size=4, fps=24):
#     num_pixels = len(bin_string)
#     pixels_per_image = (width // pixel_size) * (height // pixel_size)
#     num_images = math.ceil(num_pixels / pixels_per_image)
#     frames = []

#     for i in tqdm(range(num_images)):
#         start_index = i * pixels_per_image
#         end_index = min(start_index + pixels_per_image, num_pixels)
#         binary_digits = bin_string[start_index:end_index]

#         img = Image.new('RGB', (width, height), color='white')
#         for row_index in range(height // pixel_size):
#             start_index = row_index * (width // pixel_size)
#             end_index = start_index + (width // pixel_size)
#             row = binary_digits[start_index:end_index]

#             for col_index, digit in enumerate(row):
#                 color = (0, 0, 0) if digit == '1' else (255, 255, 255)
#                 x1, y1 = col_index * pixel_size, row_index * pixel_size
#                 img.paste(color, (x1, y1, x1 + pixel_size, y1 + pixel_size))

#         frames.append(np.array(img))

#     clip = ImageSequenceClip(frames, fps=fps)
#     clip.write_videofile('output_video.mp4', fps=fps)

# def video_to_encrypted_binary(video_path, output_file='encrypted_video.bin', encryption_key=123):
#     # Extract frames from the video
#     frames = ExtractFrames(video_path)
#     binary_data = ""

#     # Process each frame to convert it to binary
#     for frame in tqdm(frames, desc="Converting video frames to binary"):
#         gray_frame = np.mean(frame, axis=2).astype(np.uint8)
#         binary_digits = ''
#         threshold = 128  # This can be adjusted
#         for y in range(gray_frame.shape[0]):
#             for x in range(gray_frame.shape[1]):
#                 color = gray_frame[y, x]
#                 binary_digits += '1' if color < threshold else '0'
#         binary_data += binary_digits

#     # Encrypt the binary data
#     encrypted_data = encrypt_decrypt(bytearray(int(binary_data[i:i + 8], 2) for i in range(0, len(binary_data), 8)), encryption_key)

#     # Write the encrypted data to file
#     with open(output_file, 'wb') as f:
#         f.write(encrypted_data)
#     print(f"Encrypted binary data saved as {output_file}")


# def process_images(frames):
#     threshold = 128
#     binary_digits = ''

#     for frame in tqdm(frames, desc="Processing frames"):
#         gray_frame = np.mean(frame, axis=2).astype(np.uint8)

#         pixel_size = 4
#         for y in range(0, gray_frame.shape[0], pixel_size):
#             for x in range(0, gray_frame.shape[1], pixel_size):
#                 color = gray_frame[y:y + pixel_size, x:x + pixel_size]
#                 binary_digits += '1' if color.mean() < threshold else '0'

#     return binary_digits

# def binaryToFile(binary_filename, output_filename='reverse.mkv'):
#     binary_data = bytes(int(binary_filename[i:i + 8], 2) for i in range(0, len(binary_filename), 8))
#     with open(output_filename, "wb") as f:
#         f.write(binary_data)
#     print(f"Binary data converted to {output_filename}")

# def ExtractFrames(video_path):
#     frames = []
#     vid = imageio.get_reader(video_path, 'ffmpeg')
#     num_frames = vid.get_length()

#     with tqdm(total=num_frames) as pbar:
#         for i, frame in enumerate(vid):
#             frames.append(frame)
#             pbar.update(1)

#     return frames

# def image_to_encrypted_binary(image_path, encryption_key=123, output_file='encrypted_image.bin'):
#     with Image.open(image_path) as img:
#         img = img.convert('RGB')
#         binary_data = bytearray()
#         for pixel in np.array(img):
#             binary_data.extend(pixel.flatten())

#     encrypted_data = encrypt_decrypt(binary_data, encryption_key)
#     with open(output_file, 'wb') as f:
#         f.write(encrypted_data)
#     print(f"Encrypted data saved as {output_file}")

# def binary_to_image_from_file(encrypted_file, output_filename='output_image.png', decryption_key=123):
#     with open(encrypted_file, 'rb') as f:
#         encrypted_data = f.read()

#     decrypted_data = encrypt_decrypt(encrypted_data, decryption_key)
#     expected_size = 1080 * 1920 * 3  # 3 for RGB

#     if len(decrypted_data) != expected_size:
#         print(f"Warning: Expected size {expected_size} but got {len(decrypted_data)}. Cannot reconstruct image.")
#         return

#     img_data = np.frombuffer(decrypted_data, dtype=np.uint8).reshape((1080, 1920, 3))
#     img = Image.fromarray(img_data, 'RGB')
#     img.save(output_filename)
#     print(f"Image reconstructed and saved as {output_filename}")

# # Function to convert encrypted binary to video
# def encrypted_binary_to_video(encrypted_file, output_video='output_video.mp4', width=1920, height=1080, fps=24):
#     with open(encrypted_file, 'rb') as f:
#         encrypted_data = f.read()

#     decrypted_data = encrypt_decrypt(encrypted_data, 123)
#     binary_string = ''.join(format(byte, '08b') for byte in decrypted_data)
#     num_pixels = len(binary_string)

#     pixels_per_image = width * height
#     num_images = math.ceil(num_pixels / pixels_per_image)
#     frames = []

#     for i in tqdm(range(num_images)):
#         start_index = i * pixels_per_image
#         end_index = min(start_index + pixels_per_image, num_pixels)
#         binary_digits = binary_string[start_index:end_index]

#         img = Image.new('RGB', (width, height), color='white')
#         for row_index in range(height):
#             start_index = row_index * width
#             end_index = start_index + width
#             row = binary_digits[start_index:end_index]

#             for col_index, digit in enumerate(row):
#                 color = (0, 0, 0) if digit == '1' else (255, 255, 255)
#                 img.putpixel((col_index, row_index), color)

#         frames.append(np.array(img))

#     clip = ImageSequenceClip(frames, fps=fps)
#     clip.write_videofile(output_video, fps=fps)
#     print(f"Video created from encrypted binary and saved as {output_video}")

# # New functions for text to encrypted binary and encrypted binary to text
# def text_to_encrypted_binary(text_file, output_filename='encrypted_binary_text.bin', encryption_key=123):
#     with open(text_file, 'r') as f:
#         text = f.read()
#     binary_string = ''.join(format(ord(char), '08b') for char in text)

#     # Convert binary string to bytes
#     byte_array = bytearray(int(binary_string[i:i + 8], 2) for i in range(0, len(binary_string), 8))

#     # Encrypt the binary data
#     encrypted_data = encrypt_decrypt(byte_array, encryption_key)

#     with open(output_filename, 'wb') as f:
#         f.write(encrypted_data)
#     print(f"Text converted to encrypted binary and saved as {output_filename}")

# def encrypted_binary_to_text(encrypted_file, output_filename='decrypted_text.txt', decryption_key=123):
#     with open(encrypted_file, 'rb') as f:
#         encrypted_data = f.read()

#     # Decrypt the data
#     decrypted_data = encrypt_decrypt(encrypted_data, decryption_key)

#     # Convert bytes back to binary string
#     binary_string = ''.join(format(byte, '08b') for byte in decrypted_data)

#     # Convert binary string to text
#     text = ''.join(chr(int(binary_string[i:i + 8], 2)) for i in range(0, len(binary_string), 8))

#     with open(output_filename, 'w') as f:
#         f.write(text)
#     print(f"Encrypted binary converted to text and saved as {output_filename}")

# # Main menu
# if __name__ == "__main__":
#     while True:
#         input_Data = input(
#             "Choose an option:\n"
#             "1. Convert image to encrypted binary\n"
#             "2. Convert binary to file\n"
#             "3. Convert text to encrypted binary\n"
#             "4. Convert encrypted binary to text\n"
#             "5. Convert encrypted binary to video\n"
#             "6. Convert video to encrypted binary\n"
#             "Press any other key to exit.\n"
#         )


#         if input_Data == "1":
#             input_file = input("Enter the path to the image file: ")
#             image_to_encrypted_binary(input_file)

#         elif input_Data == "2":
#             binary_file = input("Enter the path to the binary file: ")
#             binaryToFile(binary_file)

#         elif input_Data == "3":
#             text_file = input("Enter the path to the text file: ")
#             text_to_encrypted_binary(text_file)

#         elif input_Data == "4":
#             encrypted_file = input("Enter the path to the encrypted binary file: ")
#             encrypted_binary_to_text(encrypted_file)

#         elif input_Data == "5":
#             encrypted_file = input("Enter the path to the encrypted binary file: ")
#             encrypted_binary_to_video(encrypted_file)

#         elif input_Data == "6":
#             video_file = input("Enter the path to the video file: ")
#             video_to_encrypted_binary(video_file)

#         else:
#             break


# ---------------------------------------------------------------------------------------
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import os
import base64
import cv2
import numpy as np

def generate_key():
    """Generate a random AES key."""
    return os.urandom(32)  # For AES-256, use a 32-byte key

def encrypt_binary(binary_data, key):
    """Encrypt binary data using AES."""
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(binary_data) + padder.finalize()
    encrypted = encryptor.update(padded_data) + encryptor.finalize()
    return base64.b64encode(iv + encrypted).decode('utf-8')

def decrypt_binary(enc_data, key):
    """Decrypt the encrypted binary data using AES."""
    enc_data_bytes = base64.b64decode(enc_data)
    iv = enc_data_bytes[:16]
    encrypted = enc_data_bytes[16:]
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_padded = decryptor.update(encrypted) + decryptor.finalize()
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    decrypted = unpadder.update(decrypted_padded) + unpadder.finalize()
    return decrypted

def encrypt_file(file_path, key):
    """Encrypt the contents of a file."""
    with open(file_path, 'rb') as f:
        binary_data = f.read()
    return encrypt_binary(binary_data, key)

def decrypt_file(enc_data, key, output_path):
    """Decrypt the encrypted data and save it to a file."""
    decrypted_data = decrypt_binary(enc_data, key)
    with open(output_path, 'wb') as f:
        f.write(decrypted_data)

def binary_to_images(binary_data, img_width=720, img_height=1080):
    """Convert binary data into a list of images with padding if needed."""
    images = []
    arr = np.frombuffer(binary_data, dtype=np.uint8)

    # Pad if necessary
    if len(arr) < img_width * img_height:
        padding_size = (img_width * img_height) - len(arr)
        pad_data = bytearray([0xAA] * padding_size)  # Using 0xAA as a padding symbol
        arr = np.concatenate((arr, pad_data))
    
    # Create images from binary data
    for i in range(0, len(arr), img_width * img_height):
        img_data = arr[i:i + img_width * img_height]
        if len(img_data) < img_width * img_height:
            break
        img_data = img_data.reshape((img_height, img_width))
        images.append(img_data)
    
    return images

def create_video_from_images(images, video_path='output_video.avi', fps=10):
    """Create a video from a list of images."""
    if not images:
        print("No images to create a video.")
        return

    height, width = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height), False)
    for img in images:
        video_writer.write(img)
    video_writer.release()
    print(f"Video saved as: {video_path}")

def extract_images_from_video(video_path):
    """Extract images from the video and return as binary data."""
    cap = cv2.VideoCapture(video_path)
    binary_data = bytearray()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        binary_data.extend(frame.flatten())
    
    cap.release()
    return bytes(binary_data)

# Example usage
if __name__ == "__main__":
    key = generate_key()

    while True:
        print("\nChoose an option:")
        print("1. Encrypt a text file")
        print("2. Decrypt a text file")
        print("3. Convert file to video")
        print("4. Extract file from video")
        print("5. Exit")

        choice = input("Enter your choice (1/2/3/4/5): ")

        if choice == '1':
            input_file_path = input("Enter the path of the text file to encrypt: ")
            encrypted_data = encrypt_file(input_file_path, key)
            output_file_path = input("Enter the path to save the encrypted file: ")
            with open(output_file_path, 'w') as enc_file:
                enc_file.write(encrypted_data)
            print(f"Encrypted Data has been saved to: {output_file_path}")

        elif choice == '2':
            enc_file_path = input("Enter the path of the encrypted file (base64): ")
            with open(enc_file_path, 'r') as enc_file:
                encrypted_data = enc_file.read()
            output_file_path = input("Enter the path where you want to save the decrypted file: ")
            decrypt_file(encrypted_data, key, output_file_path)
            print(f"Decrypted data has been saved to: {output_file_path}")

        elif choice == '3':
            file_path = input("Enter the path of the file to convert to video: ")
            binary_data = encrypt_file(file_path, key)  # Encrypt the file first
            images = binary_to_images(base64.b64decode(binary_data))  # Convert binary to images
            if not images:  # Check if images were created successfully
                print("No images generated from the binary data.")
                continue
            video_path = input("Enter the path to save the video file: ")
            create_video_from_images(images, video_path)
        
        elif choice == '4':
            video_path = input("Enter the path of the video file: ")
            binary_data = extract_images_from_video(video_path)  # Extract binary data
            output_file_path = input("Enter the path to save the extracted file: ")
            with open(output_file_path, 'wb') as f:
                f.write(binary_data)  # Save the extracted data
            print(f"Extracted data has been saved to: {output_file_path}")

        elif choice == '5':
            print("Exiting the program.")
            break

        else:
            print("Invalid choice. Please select 1, 2, 3, 4, or 5.")
