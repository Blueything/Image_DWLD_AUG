import os
import random
import threading
import requests
import pandas as pd
from io import BytesIO
from PIL import Image, ImageEnhance
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================== CONFIG ==================
ROOT_DIR = os.getcwd()          # run script in parent directory
MAX_IMAGES_PER_SPECIES = 1000
MAX_THREADS = 16
TIMEOUT = 15
IMAGE_SIZE = (512, 512)         # optional resize
# ============================================

print_lock = threading.Lock()


def log(msg):
    with print_lock:
        print(msg)


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def download_image(url, save_path):
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        img.save(save_path, "JPEG", quality=90)
        return True
    except Exception:
        return False


def augment_image(img):
    """Random lightweight augmentation"""
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    angle = random.choice([0, 90, 180, 270])
    img = img.rotate(angle)

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))

    return img


def process_species(species_path):
    species_name = os.path.basename(species_path)
    images_dir = os.path.join(species_path, "images")
    safe_mkdir(images_dir)

    log(f"\n📂 Processing species: {species_name}")

    # Collect CSVs
    csv_files = [
        os.path.join(species_path, f)
        for f in os.listdir(species_path)
        if f.endswith(".csv")
    ]

    if not csv_files:
        log(f"⚠️ No CSV found for {species_name}")
        return

    df = pd.concat([pd.read_csv(csv) for csv in csv_files], ignore_index=True)

    if "image_url" not in df.columns:
        log(f"❌ image_url column missing for {species_name}")
        return

    urls = df["image_url"].dropna().unique().tolist()
    random.shuffle(urls)
    urls = urls[:MAX_IMAGES_PER_SPECIES]

    downloaded = []
    futures = []

    log(f"⬇️ Downloading {len(urls)} images for {species_name}")

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        for idx, url in enumerate(urls, 1):
            img_name = f"BugLiteAI_{species_name}_Img{idx:04d}.jpg"
            save_path = os.path.join(images_dir, img_name)
            futures.append(executor.submit(download_image, url, save_path))

        for i, future in enumerate(as_completed(futures), 1):
            if future.result():
                downloaded.append(i)

    downloaded_count = len(downloaded)
    augment_count = 0

    log(f"✅ Downloaded {downloaded_count} images for {species_name}")

    # Augmentation if needed
    if downloaded_count < MAX_IMAGES_PER_SPECIES:
        log(f"🧪 Augmenting images for {species_name}")

        existing_images = [
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.endswith(".jpg")
        ]

        img_index = downloaded_count + 1

        while img_index <= MAX_IMAGES_PER_SPECIES:
            src_img_path = random.choice(existing_images)
            img = Image.open(src_img_path)
            img = augment_image(img)

            img_name = f"BugLiteAI_{species_name}_Img{img_index:04d}.jpg"
            img.save(os.path.join(images_dir, img_name), "JPEG", quality=90)

            augment_count += 1
            img_index += 1

    log(
        f"📊 {species_name} | Downloaded: {downloaded_count} | "
        f"Augmented: {augment_count} | Total: {MAX_IMAGES_PER_SPECIES}"
    )


def main():
    log("🚀 Starting BugLiteAI image pipeline")

    species_folders = [
        os.path.join(ROOT_DIR, d)
        for d in os.listdir(ROOT_DIR)
        if os.path.isdir(d)
    ]

    for species in species_folders:
        process_species(species)

    log("\n🎉 All species processed successfully")


if __name__ == "__main__":
    main()
