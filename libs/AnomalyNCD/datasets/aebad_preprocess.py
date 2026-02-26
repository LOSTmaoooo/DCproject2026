import sys
import os
import shutil
import glob
import cv2
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.modules._MEBin import MEBin

def aebad_crop(origin_path, save_crop_path):

    # 1: convert the original dataset to the format adapted for image cropping
    tmp_path = 'data/tmp_dataset'
    AeBAD_S_image_path = os.path.join(origin_path, 'AeBAD_S', 'test')
    AeBAD_S_mask_path = os.path.join(origin_path, 'AeBAD_S', 'ground_truth')
    anomaly_classes = sorted([d.name for d in os.scandir(AeBAD_S_mask_path) if d.is_dir()])
    for anomaly_class in anomaly_classes:
        AeBAD_S_ano_image_path = os.path.join(AeBAD_S_image_path, anomaly_class)
        AeBAD_S_ano_mask_path = os.path.join(AeBAD_S_mask_path, anomaly_class)
        tmp_ano_image_path = os.path.join(tmp_path, 'images', anomaly_class)
        os.makedirs(tmp_ano_image_path, exist_ok=True)
        tmp_ano_mask_path = os.path.join(tmp_path, 'masks', anomaly_class)
        os.makedirs(tmp_ano_mask_path, exist_ok=True)
        for image_file in glob.glob(os.path.join(AeBAD_S_ano_image_path, "**", "*.png"), recursive=True):
            shutil.copy(image_file, tmp_ano_image_path)
        for mask_file in glob.glob(os.path.join(AeBAD_S_ano_mask_path, "**", "*.png"), recursive=True):
            shutil.copy(mask_file, tmp_ano_mask_path)

    # 2: image crop
    save_max = 2
    # use the crop function in MEBin
    bin = MEBin()
    
    anomaly_classes = os.listdir(os.path.join(tmp_path, 'images'))
    for anomaly_class in anomaly_classes:
        save_path = os.path.join(save_crop_path, 'images', anomaly_class)
        os.makedirs(save_path, exist_ok=True)
        save_mask_path = os.path.join(save_crop_path, 'masks', anomaly_class)
        os.makedirs(save_mask_path, exist_ok=True)

        print('start processing ' + anomaly_class + ' data')
        
        images_path = os.path.join(tmp_path, 'images', anomaly_class)
        masks_path = os.path.join(tmp_path, 'masks', anomaly_class)
        image_files = os.listdir(images_path)
        for _, image in enumerate(tqdm(image_files)):
            image_path = os.path.join(images_path, image)
            mask_path = os.path.join(masks_path, image)

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            result, mask_result = bin.crop_sub_image_mask(image=image, mask=mask, anomaly_map=None, est_anomaly_num=save_max-1)
            
            prefix = image_path.split('.')[0].split('/')[-1]
            for i,img in enumerate(result): 
                img.save(os.path.join(save_path, "{}_crop{}.png".format(prefix, i)))
            for i,img in enumerate(mask_result): 
                img.save(os.path.join(save_mask_path, "{}_crop{}.png".format(prefix, i)))

    shutil.rmtree(tmp_path)

if __name__ == "__main__":
    origin_path = 'data/AeBAD'
    save_crop_path = 'data/AeBAD_crop'
    aebad_crop(origin_path, save_crop_path)