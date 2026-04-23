from pathlib import Path

# CHANGE THIS
images_dir = Path(r"C:\Users\vilja\Desktop\detection\a\false")
labels_dir = Path(r"C:\Users\vilja\Desktop\detection\a\false_labels")

labels_dir.mkdir(parents=True, exist_ok=True)

image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}

count = 0
for img in images_dir.iterdir():
    if img.suffix.lower() in image_exts:
        label_path = labels_dir / f"{img.stem}.txt"
        if not label_path.exists():
            label_path.write_text("")  # empty file
            count += 1

print(f"Created {count} empty label files.")