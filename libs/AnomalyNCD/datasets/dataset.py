import os
import numpy as np
from copy import deepcopy
import numpy as np

import PIL
import torch


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]



class Dataset_AnomalyNCD(torch.utils.data.Dataset):
    """
    PyTorch Dataset for AnomalyNCD.
    """

    def __init__(
        self,
        source,
        base_path,
        novel_class,
        transform=None,
        target_transform=None,
        imagesize=224,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the AnomalyNCD data folder.
            base_path: [str]. Path to the base data folder.
            novel_class: [str]. Name of unlabeled class in this dataset.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            transform: [torchvision.transforms]. Transform to apply to the training data.
            target_transform: [torchvision.transforms]. Transform to apply to the target data.
        """
        super().__init__()
        self.source = source
        self.base_path = base_path
        self.base_class = os.path.basename(base_path)
        self.novel_class = novel_class

        self.transform = transform
        self.target_transform = target_transform

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.uq_idxs = np.array(range(len(self)))

        self.imagesize = (3, imagesize, imagesize)


    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        mask = PIL.Image.open(mask_path).convert("L")

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        if self.target_transform is not None:
            target = self.target_transform(anomaly)

        return image, target, self.uq_idxs[idx], image_path, mask, mask_path
        

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        
        imgpaths_per_class = {}
        maskpaths_per_class = {}
        imgpaths_per_class[self.novel_class] = {}
        imgpaths_per_class[self.base_class] = {}
        maskpaths_per_class[self.novel_class] = {}
        maskpaths_per_class[self.base_class] = {}

        # load base_class
        classpath = os.path.join(self.base_path, 'images')
        maskpath = os.path.join(self.base_path, 'masks')
        anomaly_types = os.listdir(classpath)

        for anomaly in anomaly_types:
            anomaly_path = os.path.join(classpath, anomaly)
            anomaly_files = sorted(os.listdir(anomaly_path))
            imgpaths_per_class[self.base_class][anomaly] = [                                      
                os.path.join(anomaly_path, x) for x in anomaly_files
            ]

            anomaly_mask_path = os.path.join(maskpath, anomaly)
            anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
            maskpaths_per_class[self.base_class][anomaly] = [
                    os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files]


        # load novel_class
        classpath = os.path.join(self.source, self.novel_class, 'images')
        maskpath = os.path.join(self.source, self.novel_class, 'masks')
        anomaly_types = os.listdir(classpath)

        for anomaly in anomaly_types:
            anomaly_path = os.path.join(classpath, anomaly)
            anomaly_files = sorted(os.listdir(anomaly_path))
            imgpaths_per_class[self.novel_class][anomaly] = [                                      
                os.path.join(anomaly_path, x) for x in anomaly_files
            ]

            anomaly_mask_path = os.path.join(maskpath, anomaly)
            anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
            maskpaths_per_class[self.novel_class][anomaly] = [
                    os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files]


        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in [self.base_class, self.novel_class]:
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate
    

def subsample_dataset(dataset, idxs):
    """
    Subsample a dataset given a indice list
    """
    if len(idxs) > 0:
        mask = np.zeros(len(dataset)).astype('bool')
        mask[idxs] = True
        dataset.data_to_iterate = [data for data, mask in zip(dataset.data_to_iterate, mask) if mask] 
        dataset.uq_idxs = dataset.uq_idxs[mask]
        return dataset
    else:
        return None
    

def subsample_classes(dataset, include_classes):
    """
    Subsample a dataset given a list of classes
    """
    
    cls_idxs = [idx for idx, data_tuple in enumerate(dataset.data_to_iterate) if data_tuple[1] in include_classes]
    
    dataset = subsample_dataset(dataset, cls_idxs)

    return dataset


def get_anomalyncd_datasets(train_transform, 
                                test_transform, 
                                base_path, 
                                category,
                                labelled_classes, 
                                unlabelled_classes, 
                                data_root, 
                                seed=0):
    """
    Args:
        train_transform: [torchvision.transforms]. Transform to apply to the training data.
        test_transform: [torchvision.transforms]. Transform to apply to the test data.
        base_path: [str]. Path to the base data folder.
        category: [str]. Name of unlabeled class.
        labelled_classes: [list]. List of labeled classes.
        unlabelled_classes: [list]. List of unlabeled classes.
        data_root: [str]. Path to the AnomalyNCD data folder.
        seed: [int]. Random seed.
    Returns:
        all_datasets: [dict]. Dictionary containing the training and test datasets.
    """
    
    np.random.seed(seed)
    
    # all
    whole_set = Dataset_AnomalyNCD(source=data_root, base_path=base_path, novel_class=category, transform=train_transform)

    # label
    train_dataset_labelled = subsample_classes(deepcopy(whole_set), include_classes=labelled_classes)

    # unlabel
    train_dataset_unlabelled = subsample_classes(deepcopy(whole_set), include_classes=unlabelled_classes)

    # test
    test_dataset = subsample_classes(deepcopy(whole_set), include_classes=unlabelled_classes)                                                                                
    test_dataset.transform = test_transform

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'test': test_dataset,
    }

    return all_datasets
    