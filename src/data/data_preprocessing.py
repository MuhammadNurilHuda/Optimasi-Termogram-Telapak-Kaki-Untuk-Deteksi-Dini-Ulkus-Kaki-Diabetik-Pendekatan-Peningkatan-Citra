# src/data/data_preprocessing.py

import os
import logging
import logging.config
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
import cv2  # OpenCV untuk manipulasi citra

# Mengatur logging segera setelah impor
logging.config.fileConfig('configs/logging.conf')

# Dapatkan logger untuk modul ini
logger = logging.getLogger('src.data.data_preprocessing')

def calculate_average(directories):
    """
    Menghitung rata-rata lebar dan tinggi gambar dalam semua direktori yang diberikan.

    Parameters
    ----------
    directories : list of str
        Daftar jalur direktori yang berisi gambar-gambar.

    Returns
    -------
    avg_width : int
        Rata-rata lebar gambar.
    avg_height : int
        Rata-rata tinggi gambar.

    Raises
    ------
    FileNotFoundError
        Jika tidak ada file gambar di direktori-direktori yang diberikan.
    Exception
        Jika terjadi kesalahan lain selama proses.

    """
    widths, heights = [], []
    has_images = False
    try:
        for base_dir in directories:
            if not os.path.exists(base_dir):
                logger.warning(f"Direktori {base_dir} tidak ditemukan, melewatkan direktori ini.")
                continue
            # Iterasi melalui subdirektori
            for root, dirs, files in os.walk(base_dir):
                image_files = [
                    os.path.join(root, fname)
                    for fname in files
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
                ]
                for image_path in image_files:
                    with Image.open(image_path) as img:
                        widths.append(img.width)
                        heights.append(img.height)
                        has_images = True

        if not has_images:
            raise FileNotFoundError(f"Tidak ada file gambar di direktori-direktori yang diberikan.")

        avg_width = int(np.mean(widths))
        avg_height = int(np.mean(heights))
        logger.info(f"Rata-rata ukuran gambar: {avg_width} x {avg_height}")
        return avg_width, avg_height

    except Exception as e:
        logger.error(f"Terjadi kesalahan saat menghitung rata-rata ukuran gambar: {e}")
        raise

def resize_images(input_dir, output_dir, target_size):
    """
    Mengubah ukuran semua gambar dalam direktori input (termasuk subdirektori) ke ukuran target dan menyimpannya ke direktori output.

    Parameters
    ----------
    input_dir : str
        Jalur ke direktori input yang berisi gambar-gambar untuk diubah ukurannya.
    output_dir : str
        Jalur ke direktori output untuk menyimpan gambar yang telah diubah ukurannya.
    target_size : tuple of int
        Ukuran target dalam piksel (width, height).

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        Jika direktori input tidak ditemukan atau tidak ada gambar di dalamnya.
    Exception
        Jika terjadi kesalahan lain selama proses.

    """
    try:
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Direktori {input_dir} tidak ditemukan.")

        has_images = False
        # Iterasi melalui subdirektori
        for root, dirs, files in os.walk(input_dir):
            relative_root = os.path.relpath(root, input_dir)
            output_root = os.path.join(output_dir, relative_root)
            os.makedirs(output_root, exist_ok=True)

            for fname in files:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    has_images = True
                    input_path = os.path.join(root, fname)
                    output_path = os.path.join(output_root, fname)

                    # Mengubah ukuran dan menyimpan gambar tanpa normalisasi
                    image = cv2.imread(input_path)
                    if image is None:
                        logger.warning(f"Citra {input_path} tidak dapat dibaca, melewatkan file ini.")
                        continue
                    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
                    cv2.imwrite(output_path, resized_image)
                    logger.info(f"Menyimpan gambar yang telah diubah ukurannya ke {output_path}")

        if not has_images:
            raise FileNotFoundError(f"Tidak ada file gambar di direktori {input_dir} dan subdirektorinya.")

    except Exception as e:
        logger.error(f"Terjadi kesalahan saat mengubah ukuran gambar: {e}")
        raise

def resize_all_images(base_input_dir, base_output_dir, target_size):
    """
    Mengubah ukuran semua gambar di dalam struktur direktori yang diberikan (Left dan Right).

    Parameters
    ----------
    base_input_dir : str
        Jalur ke direktori dasar input yang berisi folder 'Left' dan 'Right'.
    base_output_dir : str
        Jalur ke direktori dasar output untuk menyimpan gambar yang telah diubah ukurannya.
    target_size : tuple of int
        Ukuran target dalam piksel (width, height).

    Returns
    -------
    None

    Raises
    ------
    Exception
        Jika terjadi kesalahan selama proses.

    """
    try:
        sides = ['Left', 'Right']
        for side in sides:
            input_side_dir = os.path.join(base_input_dir, side)
            output_side_dir = os.path.join(base_output_dir, side)
            if not os.path.exists(input_side_dir):
                logger.warning(f"Direktori {input_side_dir} tidak ditemukan, melewatkan sisi {side}.")
                continue

            # Mengubah ukuran gambar dalam direktori side (termasuk subdirektori)
            resize_images(input_side_dir, output_side_dir, target_size)
            logger.info(f"Mengubah ukuran gambar di {input_side_dir} dan menyimpan ke {output_side_dir}")

    except Exception as e:
        logger.error(f"Terjadi kesalahan saat mengubah ukuran semua gambar: {e}")
        raise

def load_tabular_data(file_path):
    """
    Memuat data tabular dari file CSV.

    Parameters
    ----------
    file_path : str
        Jalur ke file CSV yang akan dimuat.
    delimiter : str
        delimiter pada file .csv

    Returns
    -------
    data_tabular : pandas.DataFrame
        Data tabular yang telah dimuat.

    Raises
    ------
    Exception
        Jika terjadi kesalahan saat memuat data.
    """
    try:
        data_tabular = pd.read_csv(file_path, delimiter=";")
        logger.info(f"Data tabular berhasil dimuat dari {file_path}. Shape: {data_tabular.shape}")
        return data_tabular
    except Exception as e:
        logger.error(f"Gagal memuat data tabular dari {file_path}: {e}")
        raise


def convert_gender_to_numeric(data_tabular):
    """
    Mengonversi fitur 'Gender' menjadi numerik.

    Parameters
    ----------
    data_tabular : pandas.DataFrame
        Data tabular yang mengandung kolom 'Gender'.

    Returns
    -------
    data_tabular : pandas.DataFrame
        Data dengan kolom 'Gender' yang telah dikonversi.

    Raises
    ------
    Exception
        Jika terjadi kesalahan selama proses.

    """
    try:
        if 'Gender' not in data_tabular.columns:
            raise KeyError("Kolom 'Gender' tidak ditemukan dalam data_tabular.")
        data_tabular['Gender'] = data_tabular['Gender'].map({'M': 1, 'F': 0})
        logger.info("Mengonversi kolom 'Gender' menjadi numerik.")
        return data_tabular
    except Exception as e:
        logger.error(f"Terjadi kesalahan saat mengonversi 'Gender': {e}")
        raise

def create_labels(data_tabular):
    """
    Membuat label klasifikasi berdasarkan kolom 'Subject'.

    Parameters
    ----------
    data_tabular : pandas.DataFrame
        Data tabular yang mengandung kolom 'Subject'.

    Returns
    -------
    data_tabular : pandas.DataFrame
        Data dengan kolom 'label' yang ditambahkan.

    Raises
    ------
    Exception
        Jika terjadi kesalahan selama proses.

    """
    try:
        if 'Subject' not in data_tabular.columns:
            raise KeyError("Kolom 'Subject' tidak ditemukan dalam data_tabular.")
        labels = data_tabular['Subject'].apply(lambda x: 1 if str(x).startswith('DM') else 0)
        data_tabular['label'] = labels
        logger.info("Membuat label klasifikasi di kolom 'label'.")
        return data_tabular
    except Exception as e:
        logger.error(f"Terjadi kesalahan saat membuat label: {e}")
        raise

def normalize_tabular_data(data_tabular, features):
    """
    Menormalisasi data tabular pada fitur numerik.

    Parameters
    ----------
    data_tabular : pandas.DataFrame
        Data tabular yang mengandung fitur-fitur untuk dinormalisasi.
    features : list of str
        Daftar nama kolom fitur numerik yang akan dinormalisasi.

    Returns
    -------
    data_normalized : pandas.DataFrame
        Data tabular dengan fitur-fitur yang telah dinormalisasi.
    scaler : sklearn.preprocessing.StandardScaler
        Objek scaler yang di-fit pada data, dapat digunakan untuk transform data lain.

    Raises
    ------
    Exception
        Jika terjadi kesalahan selama proses.

    """
    try:
        scaler = StandardScaler()
        data_numeric = data_tabular[features]
        data_normalized = scaler.fit_transform(data_numeric)
        data_normalized = pd.DataFrame(data_normalized, columns=features)
        data_tabular.update(data_normalized)
        logger.info(f"Menormalisasi data tabular pada fitur: {features}")
        return data_tabular, scaler
    except Exception as e:
        logger.error(f"Terjadi kesalahan saat menormalisasi data tabular: {e}")
        raise

if __name__ == "__main__":
    # Jalur direktori input dan output
    base_input_dir = './data/processed/images_per_part/'
    base_output_dir = './data/processed/resized_images/'

    # Daftar direktori Left dan Right
    left_dir = os.path.join(base_input_dir, 'Left')
    right_dir = os.path.join(base_input_dir, 'Right')
    directories = [left_dir, right_dir]

    # Menghitung rata-rata ukuran gambar secara keseluruhan
    avg_width, avg_height = calculate_average(directories)
    target_size = (avg_width, avg_height)

    logger.info(f"Ukuran target untuk resizing: {target_size}")

    # Mengubah ukuran semua gambar
    resize_all_images(base_input_dir, base_output_dir, target_size)

    # Memuat data tabular
    tabular_data_path = './data/external/Plantar Thermogram Data Analysis.csv'
    try:
        data_tabular = load_tabular_data(tabular_data_path)
    except Exception as e:
        logger.error(f"Tidak dapat melanjutkan karena gagal memuat data tabular: {e}")
        exit(1)  # Menghentikan eksekusi jika data tidak dapat dimuat

    # Mengonversi 'Gender' menjadi numerik
    data_tabular = convert_gender_to_numeric(data_tabular)

    # Membuat label
    data_tabular = create_labels(data_tabular)

    # Menormalisasi fitur numerik
    numeric_features = [
        'General_right', 'LCA_right', 'LPA_right', 'MCA_right', 'MPA_right', 'TCI_right', 
        'General_left', 'LCA_left', 'LPA_left', 'MCA_left', 'MPA_left', 'TCI_left']
    data_tabular, scaler = normalize_tabular_data(data_tabular, numeric_features)

    # Menyimpan data tabular yang telah diproses
    data_tabular.to_csv('./data/processed/[Preprocessed]Plantar Thermogram Data Analysis.csv', index=False)
    logger.info("Data tabular telah diproses dan disimpan.")

    # Simpan scaler untuk digunakan pada data baru atau data test
    import joblib
    joblib.dump(scaler, './src/models/tabular_scaler.joblib')
    logger.info("Scaler untuk data tabular telah disimpan.")