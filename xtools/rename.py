import os

# Root directory
root_dir = r"C:\Users\vilja\Desktop\Training\train"

# Supported image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp', ".avif")

for subfolder in os.listdir(root_dir):
    subfolder_path = os.path.join(root_dir, subfolder)

    # Check if it's a directory
    if os.path.isdir(subfolder_path):
        files = os.listdir(subfolder_path)

        # Filter image files
        images = [f for f in files if f.lower().endswith(image_extensions)]

        # Sort to keep consistent order
        images.sort()

        for i, filename in enumerate(images, start=1):
            old_path = os.path.join(subfolder_path, filename)

            # Keep original extension
            ext = os.path.splitext(filename)[1]

            new_filename = f"{subfolder}_{i}{ext}"
            new_path = os.path.join(subfolder_path, new_filename)

            os.rename(old_path, new_path)

        print(f"Renamed {len(images)} images in folder: {subfolder}")

print("Done!")