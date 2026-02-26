import math
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

    
        
class rect:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.area = width * height
        self.center = [x + (width / 2), y + (height / 2)]


def min_distance_of_rectangles(rect1, rect2):
    """
    Calculate the minimum distance between two rectangles
    Args:
        rect1: a rectangle with x, y, width, height
        rect2: a rectangle with x, y, width, height
    Returns:
        min_dist: the minimum distance between two rectangles
    """
    min_dist = 0
    Dx = abs(rect1.center[0] - rect2.center[0])
    Dy = abs(rect1.center[1] - rect2.center[1])

    if (Dx < ((rect1.width + rect2.width)/ 2)) and (Dy >= ((rect1.height + rect2.height) / 2)):
        min_dist = Dy - ((rect1.height + rect2.height) / 2)

    elif (Dx >= ((rect1.width + rect2.width)/ 2)) and (Dy < ((rect1.height + rect2.height) / 2)):
        min_dist = Dx - ((rect1.width + rect2.width)/ 2)

    elif (Dx >= ((rect1.width + rect2.width)/ 2)) and (Dy >= ((rect1.height + rect2.height) / 2)):
        delta_x = Dx - ((rect1.width + rect2.width)/ 2)
        delta_y = Dy - ((rect1.height + rect2.height)/ 2)
        min_dist = math.sqrt(delta_x * delta_x  + delta_y * delta_y)

    else:
        min_dist = -1

    return min_dist
        
        
        
class MEBin:
    def __init__(self, args=None, anomaly_map_path_list=None):
        '''
        Get the anomaly maps and the threshold range.
        Args:
            args: The arguments parsed from the configuration file.
            anomaly_map_path_list: The paths of the anomaly score images to be binarized.
        '''
        if args and anomaly_map_path_list:
            self.args = args
            self.anomaly_map_path_list = anomaly_map_path_list if anomaly_map_path_list else []
            self.anomaly_map_list = [cv2.imread(x, cv2.IMREAD_GRAYSCALE) for x in self.anomaly_map_path_list]

            # Adaptively determine the threshold search range
            self.max_th, self.min_th = self.get_search_range()

        
    def get_search_range(self):
        '''
        Determine the threshold search range based on the maximum and minimum anomaly scores of the anomaly score images to be binarized, 
        as well as parameters from the configuration file.
        '''
        # Get the anomaly scores of all anomaly maps
        anomaly_score_list = [np.max(x) for x in self.anomaly_map_list]

        # Select the maximum and minimum anomaly scores from images
        max_score, min_score = max(anomaly_score_list), min(anomaly_score_list)
        max_th = max_score
        min_th = min_score
        return max_th, min_th
    

    def get_threshold(self, anomaly_num_sequence, min_interval_len):
        '''
        Find the 'stable interval' in the anomaly region number sequence.
        Stable Interval: A continuous threshold range in which the number of connected components remains constant, 
        and the length of the threshold range is greater than or equal to the given length threshold (min_interval_len).
        Args:
            anomaly_num_sequence: The number of connected components in the binarized map at each threshold.
            min_interval_len: The minimum length of the stable interval.
        '''
        interval_result = {}
        current_index = 0
        while current_index < len(anomaly_num_sequence):
            end = current_index 

            start = end 

            # Find the interval where the connected component count remains constant.
            if len(set(anomaly_num_sequence[start:end+1])) == 1 and anomaly_num_sequence[start] != 0:
                # Move the 'end' pointer forward until a different connected component number is encountered.
                while end < len(anomaly_num_sequence)-1 and anomaly_num_sequence[end] == anomaly_num_sequence[end+1]:
                    end += 1
                    current_index += 1
                # If the length of the current stable interval is greater than or equal to the given threshold (min_interval_len), record this interval.
                if end - start + 1 >= min_interval_len:
                    if anomaly_num_sequence[start] not in interval_result:
                        interval_result[anomaly_num_sequence[start]] = [(start, end)]
                    else:
                        interval_result[anomaly_num_sequence[start]].append((start, end))
            current_index += 1

        '''
        If a 'stable interval' exists, calculate the final threshold based on the longest stable interval.
        If no stable interval is found, it indicates that no anomaly regions exist, and 255 is returned.
        '''

        if interval_result:
            # Iterate through the stable intervals, calculating their lengths and corresponding number of connected component.
            count_result = {}
            for anomaly_num in interval_result:
                count_result[anomaly_num] = max([x[1] - x[0] for x in interval_result[anomaly_num]])
            est_anomaly_num = max(count_result, key=count_result.get)
            est_anomaly_num_interval_result = interval_result[est_anomaly_num]

            # Find the longest stable interval.
            longest_interval = sorted(est_anomaly_num_interval_result, key=lambda x: x[1] - x[0])[-1]

            # Use the endpoint threshold of the longest stable interval as the final threshold.
            index = longest_interval[1]
            threshold = 255 - index * self.args.sample_rate
            threshold = int(threshold*(self.max_th - self.min_th)/255 + self.min_th)
            return threshold, est_anomaly_num
        else:
            return 255, 0
        
        
    def bin_and_erode(self, anomaly_map, threshold):
        '''
        Binarize the anomaly map based on the given threshold.
        Apply erosion operation to the binarized result to reduce noise, as specified in the configuration file.
        Args:
            anomaly_map: The anomaly map to be binarized.
            threshold: The threshold used for binarization.
        '''
        bin_result = np.where(anomaly_map > threshold, 255, 0).astype(np.uint8)

        # Apply erosion operation to the binarized result
        if self.args.erode:
            kernel_size = 6
            iter_num = 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            bin_result = cv2.erode(bin_result, kernel, iterations=iter_num)
        return bin_result
    

    def binarize_anomaly_maps(self):
        '''
        Perform binarization within the given threshold search range,
        count the number of connected components in the binarized results.
        Adaptively determine the threshold according to the count,
        and perform binarization on the anomaly maps.
        '''

        print('Start binarizing anomaly maps...')
        self.binarized_maps = []
        self.anomaly_num_sequence_list = []
        self.est_anomaly_num_list = []
        
        for i, anomaly_map in enumerate(tqdm(self.anomaly_map_list)):
            # Normalize the anomaly map within the given threshold search range.
            anomaly_map_norm = np.where(anomaly_map < self.min_th, 0, ((anomaly_map - self.min_th) / (self.max_th - self.min_th)) * 255)
            anomaly_num_sequence = []

            # Search for the threshold from high to low within the given range using the specified sampling rate.
            for score in range(255, 0, -self.args.sample_rate):
                bin_result = self.bin_and_erode(anomaly_map_norm, score)
                num_labels, *rest = cv2.connectedComponentsWithStats(bin_result, connectivity=8)
                anomaly_num = num_labels - 1
                anomaly_num_sequence.append(anomaly_num)

            # Adaptively determine the threshold based on the anomaly connected component count sequence.
            threshold, est_anomaly_num = self.get_threshold(anomaly_num_sequence, self.args.min_interval_len)
            anomaly_num_sequence.append(1)

            # Binarize the anomaly image based on the determined threshold.
            bin_result = self.bin_and_erode(anomaly_map, threshold)
            self.binarized_maps.append(bin_result)
            self.anomaly_num_sequence_list.append(anomaly_num_sequence)
            self.est_anomaly_num_list.append(est_anomaly_num)

        return self.binarized_maps, self.est_anomaly_num_list
    
    
    def merge_crop_boxes(self, crop_box_list, image_shape, max_merge_dist):
        '''
        Merge the crop boxes that are close to each other.
        Args:
            crop_box_list: The list of crop boxes to be merged. Each crop box is represented as a tuple (left, up, right, bottom).
            image_shape: The shape of the image.
            max_merge_dist: The maximum distance threshold for merging.
        '''

        # set the distance threshold for merging, which is 1% of the maximum image size
        distance_threshold = int(max(image_shape) * max_merge_dist)
        merge_dict = {box: [] for box in crop_box_list}

        # calculate the other crop boxes that need to be merged for each crop box
        for box1 in crop_box_list:
            merge_dict[box1] = [box2 for box2 in crop_box_list if min_distance_of_rectangles(rect(box1[0], box1[1], box1[2] - box1[0], box1[3] - box1[1]),
                                    rect(box2[0], box2[1], box2[2] - box2[0], box2[3] - box2[1])) < distance_threshold]

        # generate the merge list
        merge_list = []
        
        for merge_group in merge_dict.values():
            if merge_group and sorted(merge_group) not in merge_list:
                merge_list.append(sorted(merge_group))


        # temporary storage for the merge groups that need to be removed
        temp_merge_list = []

        # find the merge groups that are subsets of other merge groups
        for group1 in merge_list:
            for group2 in merge_list:
                if group1 != group2 and set(group1).issubset(set(group2)):
                    temp_merge_list.append(group1)        

        # remove the subset merge groups from the merge list
        for group_to_remove in temp_merge_list:
            if group_to_remove in merge_list:
                merge_list.remove(group_to_remove)

        # traverse each merge group and merge the bounding boxes
        for merge_group in merge_list:
            # remove the first bounding box in the merge group
            if merge_group[0] in crop_box_list:
                crop_box_list.remove(merge_group[0])

            # remove the other bounding boxes in the merge group
            for other_box in merge_group[1:]:
                if other_box in crop_box_list:
                    crop_box_list.remove(other_box)

            new_merged_box = merge_group[0]

            # update the coordinates of the new bounding box
            for other_box in merge_group[1:]:
                new_merged_box = (
                    min(new_merged_box[0], other_box[0]),
                    min(new_merged_box[1], other_box[1]),
                    max(new_merged_box[2], other_box[2]),
                    max(new_merged_box[3], other_box[3])
                )

                # adjust the merged box to a square
                width = new_merged_box[2] - new_merged_box[0]
                height = new_merged_box[3] - new_merged_box[1]
                if width > height:
                    difference = (width - height) / 2
                    new_merged_box = (
                        new_merged_box[0],
                        new_merged_box[1] - difference,
                        new_merged_box[2],
                        new_merged_box[3] + difference
                    )
                elif width < height:
                    difference = (height - width) / 2
                    new_merged_box = (
                        new_merged_box[0] - difference,
                        new_merged_box[1],
                        new_merged_box[2] + difference,
                        new_merged_box[3]
                    )

            crop_box_list.append(new_merged_box) 
        
        merged_box_list = sorted(crop_box_list, key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)
        return merged_box_list


    def crop_sub_image_mask(self, image, mask, anomaly_map=None, est_anomaly_num=None, padding=0.1, max_merge_dist=0.01, min_crop_size=0.1):
        '''
        Crop the sub-images based on the mask
        Args:
            image: The original image.
            mask: The MEBin results of mask.
            anomaly_map: The anomaly map of the image.
            est_anomaly_num: The estimated number of anomalies in the image.
            padding: The padding ratio of the crop box.
            max_merge_dist: The maximum distance threshold for merging.
            min_crop_size: The minimum size ratio of the crop box.
        '''

        image_shape = image.shape
        # resize the mask to the same size as the image
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)

        # get the coordinates of the bounding box
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_list = []
        point_list = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)  
            area_list.append(h*w)
            point_list.append([x, y, w, h])
    
        # get the crop box
        if len(area_list) == 0:
            crop_box_list = [(0, 0, image_shape[1], image_shape[0])]
        else:
            area_list, sort_p_list = zip(*sorted(list(zip(area_list, point_list))))
            crop_box_list = []

            for i, point in enumerate(sort_p_list):
                x, y, width, height = point[0], point[1], point[2], point[3]
                p1, p2 = [y, x], [y+height, x+width]   
                center = [int((i+j)*0.5) for i, j in zip(p1, p2)]
                radius = max( int(max(height, width) * (1+padding) * 0.5), int(max(image_shape) * min_crop_size))   # the crop box's "radius" should be at least "min_crop_size" of the image size
                radius = min(radius, int(max(image_shape) * 0.5)) # the crop box should be within the image
                left, up, right, bottom = center[1]-radius, center[0]-radius, center[1]+radius, center[0]+radius

                # the crop box should be within the image
                if left<0:
                    left, right = 0, 2*radius
                elif right>image_shape[1]:
                    left, right = image_shape[1]-2*radius, image_shape[1]
                if up<0:
                    up, bottom = 0, 2*radius
                elif bottom>image_shape[0]:
                    up, bottom = image_shape[0]-2*radius, image_shape[0]
                crop_box_list.append((left, up, right, bottom))

        # merge the crop boxes
        merged_box_list = self.merge_crop_boxes(crop_box_list, image_shape, max_merge_dist)

        # only keep the top "est_anomaly_num" crop boxes, the est_anomaly_num is estimated by the MEBin
        if len(merged_box_list) > est_anomaly_num:
            merged_box_list = merged_box_list[:est_anomaly_num+1]

        if anomaly_map is None:
            image = Image.fromarray(image).convert("RGB")
            mask = Image.fromarray(mask).convert("L")
            image_crop_result = [image.crop(crop_box) for crop_box in merged_box_list]
            mask_crop_result = [mask.crop(crop_box) for crop_box in merged_box_list]
            return image_crop_result, mask_crop_result
        
        else:
            image = Image.fromarray(image).convert("RGB")
            mask = Image.fromarray(mask).convert("1")
            anomaly_map_norm = np.where(anomaly_map < self.min_th, 0, ((anomaly_map - self.min_th) / (self.max_th - self.min_th)) * 255)
            anomaly_map_norm = Image.fromarray(anomaly_map_norm)
            anomaly_map_resized = anomaly_map_norm.resize((image.size[0], image.size[1]))
            image_crop_result = [image.crop(crop_box) for crop_box in merged_box_list]
            mask_crop_result = [mask.crop(crop_box) for crop_box in merged_box_list]
            anomaly_map_crop_result = [np.max(np.array(anomaly_map_resized.crop(crop_box))) for crop_box in merged_box_list]
            return image_crop_result, mask_crop_result, anomaly_map_crop_result 