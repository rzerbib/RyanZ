from pathlib import Path
from typing import List, Optional

import rasterio
from torch.utils.data import Dataset
import torch
import numpy as np
#from torch.utils.data.dataset import T_co
import glob
import warnings
from .utils import get_means_stds_missing_values, get_indices_of_degree_features
import torchvision.transforms.functional as TF
import h5py
from datetime import datetime


class FireSpreadDataset(Dataset):
    def __init__(self, data_dir: str, included_fire_years: List[int], n_leading_observations: int,
                    crop_side_length: int, load_from_hdf5: bool, is_train: bool, remove_duplicate_features: bool,
                    stats_years: List[int], n_leading_observations_test_adjustment: Optional[int] = None, 
                    features_to_keep: Optional[List[int]] = None, return_doy: bool = False, transform = None):
        
        
        
        
        """_summary_

        Args:
            data_dir (str): _description_ Root directory of the dataset, should contain several folders, each corresponding to a different fire.
            included_fire_years (List[int]): _description_ Years in dataset_root that should be used in this instance of the dataset.
            n_leading_observations (int): _description_ Number of days to use as input observation. 
            crop_side_length (int): _description_ The side length of the random square crops that are computed during training and validation.
            load_from_hdf5 (bool): _description_ If True, load data from HDF5 files instead of TIF. 
            is_train (bool): _description_ Whether this dataset is used for training or not. If True, apply geometric data augmentations. If False, only apply center crop to get the required dimensions.
            remove_duplicate_features (bool): _description_ Remove duplicate static features from all time steps but the last one. Requires flattening the temporal dimension, since after removal, the number of features is not the same across time steps anymore.
            stats_years (List[int]): _description_ Which years to use for computing the mean and standard deviation of each feature. This is important for the test set, which should be standardized using the same statistics as the training set.
            n_leading_observations_test_adjustment (Optional[int], optional): _description_. Adjust the test set to look like it would with n_leading_observations set to this value. 
        In practice, this means that if n_leading_observations is smaller than this value, some samples are skipped. Defaults to None. If None, nothing is skipped. This is especially used for the train and val set. 
            features_to_keep (Optional[List[int]], optional): _description_. List of feature indices from 0 to 39, indicating which features to keep. Defaults to None, which means using all features.
            return_doy (bool, optional): _description_. Return the day of the year per time step, as an additional feature. Defaults to False.

        Raises:
            ValueError: _description_ Raised if input values are not in the expected ranges.
        """
        super().__init__()

        self.transform = transform  # <-- store it

        self.stats_years = stats_years
        self.return_doy = return_doy
        self.features_to_keep = features_to_keep
        self.remove_duplicate_features = remove_duplicate_features
        self.is_train = is_train
        self.load_from_hdf5 = load_from_hdf5
        self.crop_side_length = crop_side_length
        self.n_leading_observations = n_leading_observations
        self.n_leading_observations_test_adjustment = n_leading_observations_test_adjustment
        self.included_fire_years = included_fire_years
        self.data_dir = data_dir


        self.validate_inputs()

        # Compute how many samples to skip in the test set, to make it look like it would with n_leading_observations set to this value.
        if self.n_leading_observations_test_adjustment is None:
            self.skip_initial_samples = 0
        else:
            self.skip_initial_samples = self.n_leading_observations_test_adjustment - self.n_leading_observations
            if self.skip_initial_samples < 0:
                raise ValueError(f"n_leading_observations_test_adjustment must be greater than or equal to n_leading_observations, but got {self.n_leading_observations_test_adjustment=} and {self.n_leading_observations=}")

        # Create an inventory of all images in the dataset, and how many data points each fire contains. Since we have multiple data points per fire,
        # we need to know how many data points each fire contains, to be able to map a dataset index to a specific fire.
        self.imgs_per_fire = self.read_list_of_images()
        self.datapoints_per_fire = self.compute_datapoints_per_fire()
        self.length = sum([sum(self.datapoints_per_fire[fire_year].values())
                            for fire_year in self.datapoints_per_fire])

        # Used in preprocessing and normalization. Better to define it once than build/call for every data point
        # The one-hot matrix is used for one-hot encoding of land cover classes
        self.one_hot_matrix = torch.eye(17)
        self.means, self.stds, _ = get_means_stds_missing_values(self.stats_years)
        self.means = self.means[None, :, None, None]
        self.stds = self.stds[None, :, None, None]
        self.indices_of_degree_features = get_indices_of_degree_features()
        self.debug_log = []
        self.NaN_stacking_debug = []


    def find_image_index_from_dataset_index(self, target_id) -> (int, str, int):
        """_summary_ Given the index of a data point in the dataset, find the corresponding fire that contains it, 
        and its index within that fire.

        Args:
            target_id (_type_): _description_ Dataset index of the data point.

        Raises:
            RuntimeError: _description_ Raised if the dataset index is out of range.

        Returns:
            (int, str, int): _description_ Year, name of fire, index of data point within fire.
        """

        # Handle negative indexing, e.g. -1 should be the last item in the dataset
        if target_id < 0:
            target_id = self.length + target_id
        if target_id >= self.length:
            raise RuntimeError(
                f"Tried to access item {target_id}, but maximum index is {self.length - 1}.")

        # The index is relative to the length of the full dataset. However, we need to make sure that we know which
        # specific fire the queried index belongs to. We know how many data points each fire contains from
        # self.datapoints_per_fire.
        first_id_in_current_fire = 0
        found_fire_year = None
        found_fire_name = None
        for fire_year in self.datapoints_per_fire:
            if found_fire_year is None:
                for fire_name, datapoints_in_fire in self.datapoints_per_fire[fire_year].items():
                    if target_id - first_id_in_current_fire < datapoints_in_fire:
                        found_fire_year = fire_year
                        found_fire_name = fire_name
                        break
                    else:
                        first_id_in_current_fire += datapoints_in_fire

        in_fire_index = target_id - first_id_in_current_fire

        return found_fire_year, found_fire_name, in_fire_index

    def load_imgs(self, found_fire_year, found_fire_name, in_fire_index):
        """_summary_ Load the images corresponding to the specified data point from disk.

        Args:
            found_fire_year (_type_): _description_ Year of the fire that contains the data point.
            found_fire_name (_type_): _description_ Name of the fire that contains the data point.
            in_fire_index (_type_): _description_ Index of the data point within the fire.

        Returns:
            _type_: _description_ (x,y) or (x,y,doy) tuple, depending on whether return_doy is True or False. 
            x is a tensor of shape (n_leading_observations, n_features, height, width), containing the input data. 
            y is a tensor of shape (height, width) containing the binary next day's active fire mask.
            doy is a tensor of shape (n_leading_observations) containing the day of the year for each observation.
        """

        in_fire_index += self.skip_initial_samples
        end_index = (in_fire_index + self.n_leading_observations + 1)

        if self.load_from_hdf5:
            hdf5_path = self.imgs_per_fire[found_fire_year][found_fire_name][0]
            with h5py.File(hdf5_path, 'r') as f:
                imgs = f["data"][in_fire_index:end_index]
                
                #raw data nan debug
                self.NaN_stacking_debug.append(f"imgs[0]: {imgs[0]}")
                self.NaN_stacking_debug.append(f"img.shape: {imgs.shape}")
                
                if np.isnan(imgs).any():
                    self.debug_log.append(f"NaNs detected in RAW IMAGE: {hdf5_path}")
                
                
                
                if self.return_doy:
                    doys = f["data"].attrs["img_dates"][in_fire_index:(
                        end_index-1)]
                    doys = self.img_dates_to_doys(doys)
                    doys = torch.Tensor(doys)
            x, y = np.split(imgs, [-1], axis=0)
            # Last image's active fire mask is used as label, rest is input data
            y = y[0, -1, ...]


            self.NaN_stacking_debug.append(f"x[0]: {x[0]}")
        else:
            imgs_to_load = self.imgs_per_fire[found_fire_year][found_fire_name][in_fire_index:end_index]
            imgs = []
            for img_path in imgs_to_load:
                with rasterio.open(img_path, 'r') as ds:
                    

                    #raw data nan debug
                    img = ds.read()
                    if np.isnan(img).any():
                        self.debug_log.append(f"NaNs detected in RAW IMAGE: {img_path}")
            
                    imgs.append(img)


            
            self.NaN_stacking_debug.append(f"img[0]: {img[0]}")
            self.NaN_stacking_debug.append(f"img.shape: {img.shape}")
            
            x = np.stack([imgs[-1][9], imgs[-1][14], imgs[-1][22]], axis=0)
            y = imgs[-1][-1, ...]


            
            self.NaN_stacking_debug.append(f"x[0]: {x[0]}")

            # Only write to log file if NaNs were detected
            if self.NaN_stacking_debug:
                log_filename = "NaN_stacking_debug.txt"
                with open(log_filename, "a") as log_file:
                    log_file.write(f"\n📝 NaN_stacking_debug for index {index}:\n")
                    for entry in self.NaN_stacking_debug:
                        log_file.write(str(entry) + "\n")

                print(f"✅ NaN stacking debug log saved to {log_filename}")

                # Clear the log list after writing to avoid massive accumulation
                self.NaN_stacking_debug = []

            # Save actual NumPy arrays if NaNs were found
            if np.isnan(imgs).any():
                np.save(f"NaN_debug_imgs_{index}.npy", imgs[0])  # Saves full array in binary format

            if np.isnan(x).any():
                np.save(f"NaN_debug_x_{index}.npy", x[0])  # Saves full array

            if np.isnan(y).any():
                np.save(f"NaN_debug_y_{index}.npy", y)  # Saves full array

            print(f"✅ Saved NaN debug arrays for index {index}")

        
        if self.return_doy:
            return x, y, doys

        
        #nan in final x or y debug
        if np.isnan(x).any() or np.isnan(y).any():
            self.debug_log.append(f"⚠️ NaNs detected after stacking x or y in `load_imgs()'")

        return x, y


    def __getitem__(self, index):
        """
        Retrieves a single sample (x, y) — or (x, y, doys) if self.return_doy is True — from the dataset.

        - x is originally shaped [T, F, H, W], i.e. time steps x features x height x width.
        - We flatten the (T x F) dimensions into one 'channel' dimension, resulting in [T*F, H, W].
        - That way, PSPNet sees a standard [C, H, W] input per sample.
        """

        # 1) Locate which fire-year/name/index to load
        found_fire_year, found_fire_name, in_fire_index = self.find_image_index_from_dataset_index(index)

        # 2) Actually load the data (images + label + optional doys)
        loaded_imgs = self.load_imgs(found_fire_year, found_fire_name, in_fire_index)
        
        if self.return_doy:
            x, y, doys = loaded_imgs
        else:
            x, y = loaded_imgs

        if np.isnan(x).any() or np.isnan(y).any():
            self.debug_log.append(
                f"⚠️ NaNs after stacking in `load_imgs()` → year={found_fire_year}, fire={found_fire_name}, fire_index={in_fire_index}"
            )

        # 3) Convert from NumPy → Torch 
        x = torch.from_numpy(x)  # shape: (T, F, H, W) or (F, H, W)
        y = torch.from_numpy(y).long()  # shape: (H, W)

        if torch.isnan(x).any() or torch.isnan(y).any():
            self.debug_log.append(f"⚠️ NaNs detected after converting to tensor in `__getitem__()` for index {index}")

        # 4) Apply optional preprocessing/augmentation
        if self.transform is not None:
            x, y = self.transform(x, y)
            
            if torch.isnan(x).any() or torch.isnan(y).any():
                self.debug_log.append(f"⚠️ NaNs detected after applying transform in `__getitem__()` for index {index}")

        # 5) (Optional) remove duplicate static features if needed
        if self.remove_duplicate_features and self.n_leading_observations > 1:
            x = self.flatten_and_remove_duplicate_features_(x)

        # 6) (Optional) discard unwanted features
        elif self.features_to_keep is not None:
            if len(x.shape) != 4:
                raise NotImplementedError(f"Removing features is only implemented for 4D tensors, but got shape={x.shape}.")
            x = x[:, self.features_to_keep, ...]

        # 7) Flatten time (T) + features (F) → single channel dimension
        if len(x.shape) == 4:
            T, F, H, W = x.shape
            x = x.view(T * F, H, W)  # Now shape is [C, H, W]
        else:
            raise RuntimeError(f"Expected a 4D tensor at this point, got {x.shape}.")

        # 8) Debug Logging
        #self.NaN_stacking_debug.append("Test entry")

        if self.debug_log:
            log_filename = "debug_log.txt"
            with open(log_filename, "a") as log_file:  # Append mode
                log_file.write(f"\n📝 DEBUG LOG for index {index}:\n")
                for entry in self.debug_log:
                    log_file.write(entry + "\n")

            print(f"✅ Debug log saved to {log_filename}")

        # Reset debug logs to prevent duplicates
        self.debug_log = []

        # 9) Return the data
        print(f"Sample x shape after flattening: {x.shape}, y shape: {y.shape}")  # Moved above return so it's reachable

        if self.return_doy:
            return x, y, doys

        return x, y

    '''
    def __getitem__(self, index):
        """
        Retrieves a single sample (x, y) — or (x, y, doys) if self.return_doy is True — from the dataset.

        - x is originally shaped [T, F, H, W], i.e. time steps x features x height x width.
        - We flatten the (T x F) dimensions into one 'channel' dimension, resulting in [T*F, H, W].
        - That way, PSPNet sees a standard [C, H, W] input per sample.
        """

        # 1) Locate which fire-year/name/index to load
        found_fire_year, found_fire_name, in_fire_index = self.find_image_index_from_dataset_index(index)

        # 2) Actually load the data (images + label + optional doys)
        loaded_imgs = self.load_imgs(found_fire_year, found_fire_name, in_fire_index)
        if self.return_doy:
            x, y, doys = loaded_imgs
        else:
            x, y = loaded_imgs

        if np.isnan(x).any() or np.isnan(y).any():
            self.debug_log.append(
                f"⚠️ NaNs after stacking in `load_imgs()` → year={found_fire_year}, fire={found_fire_name}, fire_index={in_fire_index}"
            )


        # 3) Convert from NumPy → Torch 
        x = torch.from_numpy(x)  # shape: (T, F, H, W) or (F, H, W)
        y = torch.from_numpy(y).long()  # shape: (H, W)


        if torch.isnan(x).any() or torch.isnan(y).any():
            self.debug_log.append(f"⚠️ NaNs detected after converting to tensor in `__getitem__()` for index {index}")

        # 3) Preprocess/augment
        #x, y = self.preprocess_and_augment(x, y)

        # NEW: If a transform is provided, apply it
        if self.transform is not None:
            # transform expected to take (x, y) => (x, y)
            x, y = self.transform(x, y)


            if torch.isnan(x).any() or torch.isnan(y).any():
                self.debug_log.append(f"⚠️ NaNs detected after applying transform in `__getitem__()` for index {index}")

        # 4) (Optional) remove duplicate static features if needed
        if self.remove_duplicate_features and self.n_leading_observations > 1:
            x = self.flatten_and_remove_duplicate_features_(x)

        # 5) (Optional) discard unwanted features via self.features_to_keep
        elif self.features_to_keep is not None:
            # Expecting x in [T, F, H, W] shape for slicing
            if len(x.shape) != 4:
                raise NotImplementedError(
                    f"Removing features is only implemented for 4D tensors, but got shape={x.shape}."
                )
            # Keep only the features you want
            x = x[:, self.features_to_keep, ...]

        # 6) Flatten time (T) + features (F) => single channel dimension
        #    So [T, F, H, W] becomes [T*F, H, W].
        if len(x.shape) == 4:
            T, F, H, W = x.shape
            x = x.view(T * F, H, W)  # Now shape is [C, H, W]
        else:
            # If x is already 3D or something else, handle accordingly or raise an error
            raise RuntimeError(f"Expected a 4D tensor at this point, got {x.shape}.")

        # 7) Return x, y (and doys if needed)
        
        
        self.NaN_stacking_debug.append("Test entry")


        if self.debug_log:
            log_filename = "debug_log.txt"
            with open(log_filename, "a") as log_file:  # Append mode
                log_file.write(f"\n📝 DEBUG LOG for index {index}:\n")
                for entry in self.debug_log:
                    log_file.write(entry + "\n")
            
            # Optional: Print confirmation that log was written
            print(f"✅ Debug log saved to {log_filename}")

        self.debug_log = []
        if self.return_doy:
            return x, y, doys
        
        
        return x, y
        print(f"Sample x shape after flattening: {x.shape}, y shape: {y.shape}")

    '''

    def __len__(self):
        return self.length

    def validate_inputs(self):
        if self.n_leading_observations < 1:
            raise ValueError("Need at least one day of observations.")
        if self.return_doy and not self.load_from_hdf5:
            raise NotImplementedError(
                "Returning day of year is only implemented for hdf5 files.")
        if self.n_leading_observations_test_adjustment is not None:
            if self.n_leading_observations_test_adjustment < self.n_leading_observations:
                raise ValueError(
                    "n_leading_observations_test_adjustment must be greater than or equal to n_leading_observations.")
            if self.n_leading_observations_test_adjustment < 1:
                raise ValueError(
                    "n_leading_observations_test_adjustment must be greater than or equal to 1. Value 1 is used for having a single observation as input.")

    def read_list_of_images(self):
        """_summary_ Create an inventory of all images in the dataset.

        Returns:
            _type_: _description_ Returns a dictionary mapping integer years to dictionaries. 
            These dictionaries map names of fires that happened within the respective year to either
            a) the corresponding list of image files (in case hdf5 files are not used) or
            b) the individual hdf5 file for each fire.
        """
        imgs_per_fire = {}
        for fire_year in self.included_fire_years:
            imgs_per_fire[fire_year] = {}

            if not self.load_from_hdf5:
                fires_in_year = glob.glob(f"{self.data_dir}/{fire_year}/*/")
                fires_in_year.sort()
                for fire_dir_path in fires_in_year:
                    fire_name = fire_dir_path.split("/")[-2]
                    fire_img_paths = glob.glob(f"{fire_dir_path}/*.tif")
                    fire_img_paths.sort()
                    
                    imgs_per_fire[fire_year][fire_name] = fire_img_paths

                    if len(fire_img_paths) == 0:
                        warnings.warn(f"In dataset preparation: Fire {fire_year}: {fire_name} contains no images.",
                                        RuntimeWarning)
            else:
                fires_in_year = glob.glob(
                    f"{self.data_dir}/{fire_year}/*.hdf5")
                fires_in_year.sort()
                for fire_hdf5 in fires_in_year:
                    fire_name = Path(fire_hdf5).stem
                    imgs_per_fire[fire_year][fire_name] = [fire_hdf5]

        return imgs_per_fire

    def compute_datapoints_per_fire(self):
        """_summary_ Compute how many data points each fire contains. This is important for mapping a dataset index to a specific fire.

        Returns:
            _type_: _description_ Returns a dictionary mapping integer years to dictionaries. 
            The dictionaries map the fire name to the number of data points in that fire.
        """
        datapoints_per_fire = {}
        for fire_year in self.imgs_per_fire:
            datapoints_per_fire[fire_year] = {}
            for fire_name, fire_imgs in self.imgs_per_fire[fire_year].items():
                if not self.load_from_hdf5:
                    n_fire_imgs = len(fire_imgs) - self.skip_initial_samples
                else:
                    # Catch error case that there's no file
                    if not fire_imgs:
                        n_fire_imgs = 0
                    else:
                        with h5py.File(fire_imgs[0], 'r') as f:
                            n_fire_imgs = len(f["data"]) - self.skip_initial_samples
                # If we have two days of observations, and a lead of one day,
                # we can only predict the second day's fire mask, based on the first day's observation
                datapoints_in_fire = n_fire_imgs - self.n_leading_observations
                if datapoints_in_fire <= 0:
                    warnings.warn(
                        f"In dataset preparation: Fire {fire_year}: {fire_name} does not contribute data points. It contains "
                        f"{len(fire_imgs)} images, which is too few for a lead of {self.n_leading_observations} observations.",
                        RuntimeWarning)
                    datapoints_per_fire[fire_year][fire_name] = 0
                else:
                    datapoints_per_fire[fire_year][fire_name] = datapoints_in_fire
        return datapoints_per_fire

    def standardize_features(self, x):
        """_summary_ Standardizes the input data, using the mean and standard deviation of each feature. 
        Some features are excluded from this, which are the degree features (e.g. wind direction), and the land cover class.
        The binary active fire mask is also excluded, since it's added after standardization.

        Args:
            x (_type_): _description_ Input data, of shape (time_steps, features, height, width)

        Returns:
            _type_: _description_ Standardized input data, of shape (time_steps, features, height, width)
        """

        x = (x - self.means) / self.stds

        return x

    def preprocess_and_augment(self, x, y):
        """_summary_ Preprocesses and augments the input data. 
        This includes: 
        1. Slight preprocessing of active fire features, if loading from TIF files.
        2. Geometric data augmentation.
        3. Applying sin to degree features, to ensure that the extreme degree values are close in feature space.
        4. Standardization of features. 
        5. Addition of the binary active fire mask, as an addition to the fire mask that indicates the time of detection. 
        6. One-hot encoding of land cover classes.

        Args:
            x (_type_): _description_ Input data, of shape (time_steps, features, height, width)
            y (_type_): _description_ Target data, next day's binary active fire mask, of shape (height, width)

        Returns:
            _type_: _description_
        """

        x, y = torch.Tensor(x), torch.Tensor(y)

        # Preprocessing that has been done in HDF files already
        if not self.load_from_hdf5:

            # Active fire masks have nans where no detections occur. In general, we want to replace NaNs with
            # the mean of the respective feature. Since the NaNs here don't represent missing values, we replace
            # them with 0 instead.
            x[:, -1, ...] = torch.nan_to_num(x[:, -1, ...], nan=0)
            y = torch.nan_to_num(y, nan=0.0)

            # Turn active fire detection time from hhmm to hh.
            x[:, -1, ...] = torch.floor_divide(x[:, -1, ...], 100)

        y = (y > 0).long()

        # Augmentation has to come before normalization, because we have to correct the angle features when we change
        # the orientation of the image.
        if self.is_train:
            x, y = self.augment(x, y)
        else:
            x, y = self.center_crop_x32(x, y)

        # Some features take values in [0,360] degrees. By applying sin, we make sure that values near 0 and 360 are
        # close in feature space, since they are also close in reality.
        x[:, self.indices_of_degree_features, ...] = torch.sin(
            torch.deg2rad(x[:, self.indices_of_degree_features, ...]))

        # Compute binary mask of active fire pixels before normalization changes what 0 means. 
        binary_af_mask = (x[:, -1:, ...] > 0).float()

        x = self.standardize_features(x)

        # Adds the binary fire mask as an additional channel to the input data.
        x = torch.cat([x, binary_af_mask], axis=1)

        # Replace NaN values with 0, thereby essentially setting them to the mean of the respective feature.
        x = torch.nan_to_num(x, nan=0.0)

        # Create land cover class one-hot encoding, put it where the land cover integer was
        new_shape = (x.shape[0], x.shape[2], x.shape[3],
                        self.one_hot_matrix.shape[0])
        # -1 because land cover classes start at 1
        landcover_classes_flattened = x[:, 16, ...].long().flatten() - 1
        landcover_encoding = self.one_hot_matrix[landcover_classes_flattened].reshape(
            new_shape).permute(0, 3, 1, 2)
        x = torch.concatenate(
            [x[:, :16, ...], landcover_encoding, x[:, 17:, ...]], dim=1)

        return x, y

    def augment(self, x, y):
        """_summary_ Applies geometric transformations: 
            1. random square cropping, preferring images with a) fire pixels in the output and b) (with much less weight) fire pixels in the input
            2. rotate by multiples of 90°
            3. flip horizontally and vertically
        Adjustment of angles is done as in https://github.com/google-research/google-research/blob/master/simulation_research/next_day_wildfire_spread/image_utils.py

        Args:
            x (_type_): _description_ Input data, of shape (time_steps, features, height, width)
            y (_type_): _description_ Target data, next day's binary active fire mask, of shape (height, width)

        Returns:
            _type_: _description_
        """

        # Need square crop to prevent rotation from creating/destroying data at the borders, due to uneven side lengths.
        # Try several crops, prefer the ones with most fire pixels in output, followed by most fire_pixels in input
        best_n_fire_pixels = -1
        best_crop = (None, None)

        for i in range(10):
            top = np.random.randint(0, x.shape[-2] - self.crop_side_length)
            left = np.random.randint(0, x.shape[-1] - self.crop_side_length)
            x_crop = TF.crop(
                x, top, left, self.crop_side_length, self.crop_side_length)
            y_crop = TF.crop(
                y, top, left, self.crop_side_length, self.crop_side_length)

            # We really care about having fire pixels in the target. But if we don't find any there,
            # we care about fire pixels in the input, to learn to predict that no new observations will be made,
            # even though previous days had active fires.
            n_fire_pixels = x_crop[:, -1, ...].mean() + \
                1000 * y_crop.float().mean()
            if n_fire_pixels > best_n_fire_pixels:
                best_n_fire_pixels = n_fire_pixels
                best_crop = (x_crop, y_crop)

        x, y = best_crop

        hflip = bool(np.random.random() > 0.5)
        vflip = bool(np.random.random() > 0.5)
        rotate = int(np.floor(np.random.random() * 4))
        if hflip:
            x = TF.hflip(x)
            y = TF.hflip(y)
            # Adjust angles
            x[:, self.indices_of_degree_features, ...] = 360 - \
                x[:, self.indices_of_degree_features, ...]

        if vflip:
            x = TF.vflip(x)
            y = TF.vflip(y)
            # Adjust angles
            x[:, self.indices_of_degree_features, ...] = (
                180 - x[:, self.indices_of_degree_features, ...]) % 360

        if rotate != 0:
            angle = rotate * 90
            x = TF.rotate(x, angle)
            y = torch.unsqueeze(y, 0)
            y = TF.rotate(y, angle)
            y = torch.squeeze(y, 0)

            # Adjust angles
            x[:, self.indices_of_degree_features, ...] = (x[:, self.indices_of_degree_features,
                                                            ...] - 90 * rotate) % 360

        return x, y

    def center_crop_x32(self, x, y):
        """_summary_ Crops the center of the image to side lengths that are a multiple of 32, 
        which the ResNet U-net architecture requires. Only used for computing the test performance.

        Args:
            x (_type_): _description_
            y (_type_): _description_

        Returns:
            _type_: _description_
        """
        T, C, H, W = x.shape
        H_new = H//32 * 32
        W_new = W//32 * 32

        x = TF.center_crop(x, (H_new, W_new))
        y = TF.center_crop(y, (H_new, W_new))
        return x, y

    def flatten_and_remove_duplicate_features_(self, x):
        """_summary_ For a simple U-Net, static and forecast features can be removed everywhere but in the last time step
        to reduce the number of features. Since that would result in different numbers of channels for different
        time steps, we flatten the temporal dimension. 
        Also discards features that we don't want to use. 

        Args:
            x (_type_): _description_ Input tensor data of shape (n_leading_observations, n_features, height, width)

        Returns:
            _type_: _description_
        """
        static_feature_ids, dynamic_feature_ids = self.get_static_and_dynamic_features_to_keep(self.features_to_keep)
        dynamic_feature_ids = torch.tensor(dynamic_feature_ids).int()

        x_dynamic_only = x[:-1, dynamic_feature_ids, :, :].flatten(start_dim=0, end_dim=1)
        x_last_day = x[-1, self.features_to_keep, ...].squeeze(0)

        return torch.cat([x_dynamic_only, x_last_day], axis=0)

    @staticmethod
    def get_static_and_dynamic_feature_ids():
        """_summary_ Returns the indices of static and dynamic features.
        Static features include topographical features and one-hot encoded land cover classes.

        Returns:
            _type_: _description_ Tuple of lists of integers, first list contains static feature indices, second list contains dynamic feature indices.
        """
        static_feature_ids = [12,13,14] + list(range(16,33))
        dynamic_feature_ids = list(range(12)) + [15] + list(range(33,40))
        return static_feature_ids, dynamic_feature_ids

    @staticmethod
    def get_static_and_dynamic_features_to_keep(features_to_keep:Optional[List[int]]):
        """_summary_ Returns the indices of static and dynamic features that should be kept, based on the input list of feature indices to keep.

        Args:
            features_to_keep (Optional[List[int]]): _description_

        Returns:
            _type_: _description_
        """
        static_features_to_keep, dynamic_features_to_keep = FireSpreadDataset.get_static_and_dynamic_feature_ids()
        
        if type(features_to_keep) == list:
            dynamic_features_to_keep = list(set(dynamic_features_to_keep) & set(features_to_keep))
            dynamic_features_to_keep.sort()

        if type(features_to_keep) == list:
            static_features_to_keep = list(set(static_features_to_keep) & set(features_to_keep))
            static_features_to_keep.sort()

        return static_features_to_keep, dynamic_features_to_keep

    @staticmethod
    def get_n_features(n_observations:int, features_to_keep:Optional[List[int]], deduplicate_static_features:bool):
        """_summary_ Computes the number of features that the dataset will have after preprocessing, 
        considering the number of input observations, which features to keep or discard, and whether to deduplicate static features.

        Args:
            n_observations (int): _description_
            features_to_keep (Optional[List[int]]): _description_
            deduplicate_static_features (bool): _description_

        Returns:
            _type_: _description_ If deduplicate_static_features is True, returns the total number of features, flattened across all time steps. 
            Otherwise, returns the number of features per time step.
        """
        static_features_to_keep, dynamic_features_to_keep = FireSpreadDataset.get_static_and_dynamic_features_to_keep(features_to_keep)

        n_static_features = len(static_features_to_keep)
        n_dynamic_features = len(dynamic_features_to_keep)
        n_all_features = n_static_features + n_dynamic_features

        # If we deduplicate static features, we remove them from all time steps but the last one.
        # The last day then gets dynamic and static features. All other days only get dynamic features. 
        n_features = (int(deduplicate_static_features)*n_dynamic_features)*(n_observations-1) + n_all_features

        return n_features


    @staticmethod
    def img_dates_to_doys(img_dates):
        """_summary_ Converts a list of date strings to day of year values.

        Args:
            img_dates (_type_): _description_ List of date strings

        Returns:
            _type_: _description_ List of day of year values
        """
        date_format = "%Y-%m-%d"
        # In old preprocessing, the dates still had a TIF file extension, which is also removed here.
        return [datetime.strptime(img_date.replace(".tif", ""), date_format).timetuple().tm_yday for img_date in img_dates]

    @staticmethod
    def map_channel_index_to_features(only_base:bool = False):
        """_summary_ Maps the channel index to the feature name.

        Returns:
            _type_: _description_
        """

        # Features before any processing
        base_feature_names = [
            'VIIRS band M11',
            'VIIRS band I2',
            'VIIRS band I1',
            'NDVI',
            'EVI2',
            'Total precipitation',
            'Wind speed',
            'Wind direction',
            'Minimum temperature',
            'Maximum temperature',
            'Energy release component',
            'Specific humidity',
            'Slope',
            'Aspect',
            'Elevation',
            'Palmer drought severity index (PDSI)',
            'Landcover class',
            'Forecast: Total precipitation',
            'Forecast: Wind speed',
            'Forecast: Wind direction',
            'Forecast: Temperature',
            'Forecast: Specific humidity',
            'Active fire']

        # Different land cover classes of feature "Landcover class"
        land_cover_classes = [
            'Land cover: Evergreen Needleleaf Forests',
            'Land cover: Evergreen Broadleaf Forests',
            'Land cover: Deciduous Needleleaf Forests',
            'Land cover: Deciduous Broadleaf Forests',
            'Land cover: Mixed Forests',
            'Land cover: Closed Shrublands',
            'Land cover: Open Shrublands',
            'Land cover: Woody Savannas',
            'Land cover: Savannas',
            'Land cover: Grasslands',
            'Land cover: Permanent Wetlands',
            'Land cover: Croplands',
            'Land cover: Urban and Built-up Lands',
            'Land cover: Cropland/Natural Vegetation Mosaics',
            'Land cover: Permanent Snow and Ice',
            'Land cover: Barren',
            'Land cover: Water Bodies']
        
        if only_base:
            # Features as in the GeoTIFF files: land cover class not expanded, no binary active fire
            return_features = base_feature_names
        else:
            # Features as used by most experiments
            return_features = base_feature_names[:16] + land_cover_classes + base_feature_names[17:] + ["Active fire (binary)"]

        return dict(enumerate(return_features))

    def get_generator_for_hdf5(self):
        """_summary_ Creates a generator that is used to turn the dataset into HDF5 files. It applies a few 
        preprocessing steps to the active fire features that need to be applied anyway, to save some computation.

        Yields:
            _type_: _description_ Generator that yields tuples of (year, fire_name, img_dates, lnglat, img_array) 
            where img_array contains all images available for the respective fire, preprocessed such 
            that active fire detection times are converted to hours. lnglat contains longitude and latitude
            of the center of the image.
        """

        for year, fires_in_year in self.imgs_per_fire.items():
            for fire_name, img_files in fires_in_year.items():
                imgs = []
                lnglat = None
                for img_path in img_files:
                    with rasterio.open(img_path, 'r') as ds:
                        imgs.append(ds.read())
                        if lnglat is None:
                            lnglat = ds.lnglat()
                x = np.stack(imgs, axis=0)

                # Get dates from filenames
                img_dates = [img_path.split("/")[-1].split("_")[0].replace(".tif", "")
                                for img_path in img_files]

                # Active fire masks have nans where no detections occur. In general, we want to replace NaNs with
                # the mean of the respective feature. Since the NaNs here don't represent missing values, we replace
                # them with 0 instead.
                x[:, -1, ...] = np.nan_to_num(x[:, -1, ...], nan=0)

                # Turn active fire detection time from hhmm to hh.
                x[:, -1, ...] = np.floor_divide(x[:, -1, ...], 100)
                yield year, fire_name, img_dates, lnglat, x