import os

# Lokalus aplankas su klasemis, kurias pervadinam.
root_dir = r"C:\Users\vilja\Desktop\Wheel_Detection\weak"

image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp', ".avif")

for subfolder in os.listdir(root_dir):
    subfolder_path = os.path.join(root_dir, subfolder)

    # Praleidziam viska, kas nera klases aplankas
    if os.path.isdir(subfolder_path):
        files = os.listdir(subfolder_path)

        # Atsifiltruojam tik image failus
        images = [f for f in files if f.lower().endswith(image_extensions)]

        # Stabilus rusiavimas, kad numeracija butu kartojama
        images.sort()

        for i, filename in enumerate(images, start=1):
            old_path = os.path.join(subfolder_path, filename)

            # Paliekam originalu failo formata.
            ext = os.path.splitext(filename)[1]

            new_filename = f"{subfolder}_{i}{ext}"
            new_path = os.path.join(subfolder_path, new_filename)

            os.rename(old_path, new_path)

        print(f"Renamed {len(images)} images in folder: {subfolder}")

print("Done!")