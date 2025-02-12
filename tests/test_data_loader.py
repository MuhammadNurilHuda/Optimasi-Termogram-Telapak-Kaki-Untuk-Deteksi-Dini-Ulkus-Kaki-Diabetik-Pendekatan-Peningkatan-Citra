# tests/test_data_loader.py

import sys
import unittest
import os
import shutil
# Menambahkan direktori proyek utama ke sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.data import organize_images


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # Membuat direktori sementara untuk pengujian
        self.test_raw_dir = 'data/test_raw'
        self.test_output_dir = 'data/test_processed'
        os.makedirs(self.test_raw_dir, exist_ok=True)
        # Tambahkan setup data uji di sini

    def tearDown(self):
        # Menghapus direktori uji setelah pengujian selesai
        shutil.rmtree(self.test_raw_dir)
        shutil.rmtree(self.test_output_dir)

    def test_organize_images(self):
        # Panggil fungsi yang akan diuji
        organize_images(self.test_raw_dir, self.test_output_dir)
        # Tambahkan assert untuk memeriksa hasil yang diharapkan


if __name__ == '__main__':
    unittest.main()