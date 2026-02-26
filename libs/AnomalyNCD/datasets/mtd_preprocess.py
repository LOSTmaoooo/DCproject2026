import shutil
import os

def mtd_convert(org_mtd_dir, train_txt, new_mtd_dir):
    # Create the required directories for the dataset: 'MT_Free' in 'train' directory,
    os.makedirs(os.path.join(new_mtd_dir, 'train', 'MT_Free'), exist_ok=True)

    # Convert the dataset to the MVTec AD structure
    for anomaly_type in sorted([d.name for d in os.scandir(org_mtd_dir) if d.is_dir()]):
        os.makedirs(os.path.join(new_mtd_dir, 'test', anomaly_type), exist_ok=True)
        os.makedirs(os.path.join(new_mtd_dir, 'ground_truth', anomaly_type), exist_ok=True)
        image_dir = os.path.join(org_mtd_dir, anomaly_type, 'Imgs')
        for image_path in os.listdir(image_dir):
            if '.jpg' in image_path:
                shutil.copy(os.path.join(image_dir, image_path), os.path.join(new_mtd_dir, 'test', anomaly_type, image_path))
            elif '.png' in image_path:
                shutil.copy(os.path.join(image_dir, image_path), os.path.join(new_mtd_dir, 'ground_truth', anomaly_type, image_path))
            else:
                assert 'error'

    # Move 80% good to train according to the train_txt
    f = open(train_txt)
    train_images = f.readlines()
    train_images = [im.replace('\n', '') for im in train_images]
    for im in train_images:
        shutil.move(os.path.join(new_mtd_dir, 'test', 'MT_Free', im), os.path.join(new_mtd_dir, 'train', 'MT_Free', im))

if __name__ == "__main__":
    org_mtd_dir = 'data/Magnetic-tile-defect-datasets'
    train_txt = 'data/Magnetic-tile-defect-datasets/mtd_train_filenames.txt'
    new_mtd_dir = 'data/mtd_anomaly_detection'
    mtd_convert(org_mtd_dir, train_txt, new_mtd_dir)