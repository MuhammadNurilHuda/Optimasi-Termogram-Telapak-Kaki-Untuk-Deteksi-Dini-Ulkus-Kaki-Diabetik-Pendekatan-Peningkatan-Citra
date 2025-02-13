# src/apply_image_enhancements.py

import os
import cv2
import sys
import logging
import logging.config
# Menambahkan direktori proyek utama ke sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.utils.image_enhancement import (
    posterize_image,
    solarize_image,
    clahe_image,
    adjust_gamma_image
)

# Mengatur logging
logging.config.fileConfig('configs/logging.conf')
logger = logging.getLogger('src.apply_image_enhancements')

def process_images(input_base_dir, output_base_dir):
    """
    Menerapkan teknik image enhancement pada gambar dalam direktori input dan menyimpan hasilnya.
    """
    try:
        # Tentukan teknik dan parameter
        enhancements = {
            'Posterize': {
                'function': posterize_image,
                'parameters': [{'bits': b} for b in [1, 2, 3]],
            },
            'Solarize': {
                'function': solarize_image,
                'parameters': [{'threshold': t} for t in [64, 128, 192]],
            },
            'CLAHE': {
                'function': clahe_image,
                'parameters': [
                    {'clip_limit': 2.0, 'tile_grid_size': (8, 8)},
                    {'clip_limit': 3.0, 'tile_grid_size': (8, 8)},
                    {'clip_limit': 3.0, 'tile_grid_size': (6, 12)},
                    {'clip_limit': 3.0, 'tile_grid_size': (16, 16)},
                ],
            },
            'Gamma': {
                'function': adjust_gamma_image,
                'parameters': [{'gamma': g} for g in [-1.5, 0.5, 1.5, 2, 5]],
            },
        }

        # Iterasi melalui setiap teknik
        for enhancement_name, enhancement_info in enhancements.items():
            function = enhancement_info['function']
            parameters_list = enhancement_info['parameters']

            for params in parameters_list:
                # Menentukan nama subdirektori berdasarkan parameter
                subdir_name = '_'.join([str(v) for v in params.values()])
                output_dir = os.path.join(output_base_dir, enhancement_name, subdir_name)

                # Iterasi melalui gambar dalam direktori input
                for root, dirs, files in os.walk(input_base_dir):
                    # Mendapatkan path relatif untuk mempertahankan struktur direktori
                    relative_root = os.path.relpath(root, input_base_dir)
                    output_root = os.path.join(output_dir, relative_root)
                    os.makedirs(output_root, exist_ok=True)

                    for file_name in files:
                        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                            input_path = os.path.join(root, file_name)
                            output_path = os.path.join(output_root, file_name)

                            try:
                                image = cv2.imread(input_path, cv2.IMREAD_COLOR)
                                if image is None:
                                    logger.warning(f"Citra {input_path} tidak dapat dibaca, melewatkan file ini.")
                                    continue

                                # Terapkan teknik image enhancement
                                enhanced_image = function(image, **params)
                                cv2.imwrite(output_path, enhanced_image)
                                logger.info(f"Menyimpan {enhancement_name} dengan parameter {params} ke {output_path}")
                            except Exception as e:
                                logger.error(f"Gagal memproses {input_path}: {e}")

    except Exception as e:
        logger.error(f"Terjadi kesalahan selama proses image enhancement: {e}")
        raise

if __name__ == "__main__":
    # Mengatur path direktori input dan output
    input_base_dir = './data/processed/resized_images/'
    output_base_dir = './data/processed/image enhancement/'

    process_images(input_base_dir, output_base_dir)