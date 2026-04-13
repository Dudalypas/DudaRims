import os

DATASET_DIR = r"C:\Users\vilja\Desktop\Training_Mixed"
SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff", ".avif"}

def is_image(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

def count_images_in_class(class_path):
    return sum(
        1 for f in os.listdir(class_path)
        if os.path.isfile(os.path.join(class_path, f)) and is_image(f)
    )

def main():
    for split in SPLITS:
        split_path = os.path.join(DATASET_DIR, split)

        if not os.path.exists(split_path):
            print(f"\n{split.upper()} folder not found: {split_path}")
            continue

        print(f"\n=== {split.upper()} ===")
        total = 0

        classes = sorted(os.listdir(split_path))
        for cls in classes:
            cls_path = os.path.join(split_path, cls)
            if not os.path.isdir(cls_path):
                continue

            count = count_images_in_class(cls_path)
            total += count
            print(f"{cls}: {count}")

        print(f"Total {split}: {total}")

if __name__ == "__main__":
    main()