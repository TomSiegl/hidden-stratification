import os

import pandas as pd
from PIL import Image
from tqdm import tqdm


def preprocess_isic_images(root):
    """Preprocesses the images."""
    raw_dir = os.path.join(root, 'raw')
    if not os.path.isdir(os.path.join(raw_dir, 'images')):
        raise FileNotFoundError('Raw ISIC images not found. Run `download_isic_images` before '
                                'calling `preprocess_isic_images`.')
    processed_dir = os.path.join(root, 'processed')

    labels_df = pd.read_csv(os.path.join(root, 'labels.csv'))
    labels_df = labels_df.set_index('Image Index')
    image_ids = labels_df.index.tolist()

    os.makedirs(os.path.join(processed_dir, 'images'), exist_ok=True)
    for image_id in tqdm(image_ids):
        out_path = os.path.join(processed_dir, 'images', f'{image_id}')
        if os.path.isfile(out_path):
            continue
        image = Image.open(os.path.join(raw_dir, 'images', f'{image_id}'))
        image = image.resize((224, 224))
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.save(out_path)


if __name__ == '__main__':
    preprocess_isic_images('./data/isic')
