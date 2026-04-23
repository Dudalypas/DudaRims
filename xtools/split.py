import os
import shutil
import random

# Greita lokali split konfiguracija, keliai hardcodinti tho, gal persidaryt paskui
SOURCE_DIR = r"C:\Users\vilja\Desktop\Training_Mixed_v2\train"
OUTPUT_DIR = r"C:\Users\vilja\Desktop\PROD_TRAIN"
VAL_COUNT = 20
TEST_COUNT = 20
SEED = 42

random.seed(SEED)

def split_data():
    classes = os.listdir(SOURCE_DIR)

    for cls in classes:
        cls_path = os.path.join(SOURCE_DIR, cls)
        if not os.path.isdir(cls_path):
            continue

        images = [
            f for f in os.listdir(cls_path)
            if os.path.isfile(os.path.join(cls_path, f))
        ]
        random.shuffle(images)

        n = len(images)
        required = VAL_COUNT + TEST_COUNT

        if n < required:
            print(f"Skipping {cls}: only {n} images, need at least {required}")
            continue

        splits = {
            "val": images[:VAL_COUNT],
            "test": images[VAL_COUNT:VAL_COUNT + TEST_COUNT],
            "train": images[VAL_COUNT + TEST_COUNT:]
        }

        for split_name, split_files in splits.items():
            split_dir = os.path.join(OUTPUT_DIR, split_name, cls)
            os.makedirs(split_dir, exist_ok=True)

            for file in split_files:
                src = os.path.join(cls_path, file)
                dst = os.path.join(split_dir, file)
                shutil.copy2(src, dst)

        print(
            f"{cls}: train={len(splits['train'])}, "
            f"val={len(splits['val'])}, test={len(splits['test'])}"
        )

    print("Done!")

if __name__ == "__main__":
    split_data()