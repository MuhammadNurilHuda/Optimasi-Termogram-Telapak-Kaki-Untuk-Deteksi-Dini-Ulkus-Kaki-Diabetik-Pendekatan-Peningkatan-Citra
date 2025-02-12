# tests/test_data_preprocessing.py

import os
import sys
import unittest
import shutil
import pandas as pd
import numpy as np
# Menambahkan direktori proyek utama ke sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.data import (
    calculate_average,
    resize_images,
    resize_all_images,
    load_tabular_data,
    convert_gender_to_numeric,
    create_labels,
    normalize_tabular_data
)
from PIL import Image

class TestDataPreprocessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Membuat direktori sementara untuk pengujian gambar
        cls.test_image_dir = 'tests/temp_images'
        cls.test_output_dir = 'tests/temp_resized_images'
        os.makedirs(cls.test_image_dir, exist_ok=True)
        os.makedirs(cls.test_output_dir, exist_ok=True)

        # Membuat beberapa gambar dummy untuk pengujian
        cls.image_sizes = [(100, 200), (150, 250), (200, 300)]
        for i, size in enumerate(cls.image_sizes):
            img = Image.new('RGB', size, color=(i*50, i*50, i*50))
            img_path = os.path.join(cls.test_image_dir, f'test_image_{i}.png')
            img.save(img_path)

        # Membuat data tabular dummy untuk pengujian
        cls.test_tabular_file = 'tests/temp_tabular_data.csv'
        data = {
            'Subject': ['DM001', 'CG002', 'DM003'],
            'Gender': ['M', 'F', 'M'],
            'Age': [60, 55, 65],
            'Weight': [70, 60, 80],
            'Height': [170, 160, 175],
            'BMI': [24.22, 23.44, 26.12],
            'General_right': [34.5, 33.0, 35.0],
            'General_left': [34.0, 33.5, 35.5]
        }
        cls.test_tabular_data = pd.DataFrame(data)
        cls.test_tabular_data.to_csv(cls.test_tabular_file, index=False, sep=';')

    @classmethod
    def tearDownClass(cls):
        # Menghapus direktori dan file sementara setelah pengujian selesai
        shutil.rmtree(cls.test_image_dir)
        shutil.rmtree(cls.test_output_dir)
        os.remove(cls.test_tabular_file)

    def test_calculate_average(self):
        # Menguji fungsi calculate_average
        avg_width, avg_height = calculate_average([self.test_image_dir])
        expected_width = int(np.mean([size[0] for size in self.image_sizes]))
        expected_height = int(np.mean([size[1] for size in self.image_sizes]))
        self.assertEqual(avg_width, expected_width)
        self.assertEqual(avg_height, expected_height)

    def test_resize_images(self):
        # Menguji fungsi resize_images
        target_size = (100, 100)
        resize_images(self.test_image_dir, self.test_output_dir, target_size)
        # Periksa apakah gambar telah diubah ukurannya
        for fname in os.listdir(self.test_output_dir):
            img_path = os.path.join(self.test_output_dir, fname)
            with Image.open(img_path) as img:
                self.assertEqual(img.size, target_size)

    def test_resize_all_images(self):
        # Membuat struktur direktori yang mirip dengan Left dan Right
        base_input_dir = 'tests/temp_images_per_part'
        os.makedirs(base_input_dir, exist_ok=True)
        left_dir = os.path.join(base_input_dir, 'Left')
        right_dir = os.path.join(base_input_dir, 'Right')
        os.makedirs(left_dir, exist_ok=True)
        os.makedirs(right_dir, exist_ok=True)
        shutil.copytree(self.test_image_dir, os.path.join(left_dir, 'CG Left'))
        shutil.copytree(self.test_image_dir, os.path.join(right_dir, 'DM Right'))

        base_output_dir = 'tests/temp_resized_images_per_part'
        target_size = (100, 100)
        resize_all_images(base_input_dir, base_output_dir, target_size)

        # Periksa apakah gambar telah diubah ukurannya
        for side in ['Left', 'Right']:
            side_dir = os.path.join(base_output_dir, side)
            for group_dir in os.listdir(side_dir):
                group_path = os.path.join(side_dir, group_dir)
                for fname in os.listdir(group_path):
                    img_path = os.path.join(group_path, fname)
                    with Image.open(img_path) as img:
                        self.assertEqual(img.size, target_size)

        # Bersihkan direktori sementara
        shutil.rmtree(base_input_dir)
        shutil.rmtree(base_output_dir)

    def test_load_tabular_data(self):
        # Menguji fungsi load_tabular_data
        data_tabular = load_tabular_data(self.test_tabular_file)
        self.assertIsInstance(data_tabular, pd.DataFrame)
        self.assertEqual(data_tabular.shape[0], 3)  # Jumlah baris
        self.assertIn('Subject', data_tabular.columns)

    def test_convert_gender_to_numeric(self):
        # Menguji fungsi convert_gender_to_numeric
        data_tabular = load_tabular_data(self.test_tabular_file)
        data_tabular = convert_gender_to_numeric(data_tabular)
        self.assertTrue(data_tabular['Gender'].isin([0, 1]).all())

    def test_create_labels(self):
        # Menguji fungsi create_labels
        data_tabular = load_tabular_data(self.test_tabular_file)
        data_tabular = create_labels(data_tabular)
        self.assertIn('label', data_tabular.columns)
        self.assertListEqual(data_tabular['label'].tolist(), [1, 0, 1])

    def test_normalize_tabular_data(self):
        # Menguji fungsi normalize_tabular_data
        data_tabular = load_tabular_data(self.test_tabular_file)
        features = ['Age', 'Weight', 'Height', 'BMI', 'General_right', 'General_left']
        data_tabular, scaler = normalize_tabular_data(data_tabular, features)
        # Periksa apakah fitur telah dinormalisasi (mean ~ 0, std ~ 1)
        data_normalized = data_tabular[features]
        means = data_normalized.mean()
        stds = data_normalized.std(ddof=0)
        for mean in means:
            self.assertAlmostEqual(mean, 0, places=6)
        for std in stds:
            self.assertAlmostEqual(std, 1, places=6)

        # Periksa apakah scaler telah di-fit
        self.assertIsNotNone(scaler)
        self.assertTrue(hasattr(scaler, 'mean_'))

if __name__ == '__main__':
    unittest.main()