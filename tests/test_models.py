# tests/test_models.py

import unittest
import os
import sys
# Menambahkan direktori proyek utama ke sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.models import (
    create_model1, 
    create_model2, 
    create_model3, 
    create_model4
    )


class TestModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ukuran input citra (misalnya, gambar berukuran 224x224 dengan 3 channel RGB)
        cls.input_shape_image = (224, 224, 3)
        # Jumlah fitur tabular (misalnya, 12 fitur)
        cls.input_shape_tabular = 12

    def test_model1_creation(self):
        model = create_model1(self.input_shape_image, self.input_shape_tabular)
        self.assertIsNotNone(model)
        self.assertEqual(len(model.inputs), 3)
        self.assertEqual(len(model.outputs), 1)

    def test_model2_creation(self):
        model = create_model2(self.input_shape_image, self.input_shape_tabular)
        self.assertIsNotNone(model)
        self.assertEqual(len(model.inputs), 3)
        self.assertEqual(len(model.outputs), 1)

    def test_model3_creation(self):
        model = create_model3(self.input_shape_image, self.input_shape_tabular)
        self.assertIsNotNone(model)
        self.assertEqual(len(model.inputs), 3)
        self.assertEqual(len(model.outputs), 1)

    def test_model4_creation(self):
        model = create_model4(self.input_shape_image, self.input_shape_tabular)
        self.assertIsNotNone(model)
        self.assertEqual(len(model.inputs), 3)
        self.assertEqual(len(model.outputs), 1)

if __name__ == '__main__':
    unittest.main()