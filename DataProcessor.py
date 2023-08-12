import os
import numpy as np
import shutil

class DataProcessor:
    """
    Class for processing and distributing data across directories.
    """

    def process_directory(self, old_dir_path, new_dir_path):
        """
        Process a directory, load data from files, merge and save to a new directory.
        
        Args:
        - old_dir_path (str): Path to the old directory
        - new_dir_path (str): Path to the new directory
        """
        # Create new directory if it doesn't exist
        os.makedirs(new_dir_path, exist_ok=True)
        
        for folder in os.listdir(old_dir_path):
            old_folder_path = os.path.join(old_dir_path, folder)
            new_folder_path = os.path.join(new_dir_path, folder)

            # Create a sub-folder in the new directory
            os.makedirs(new_folder_path, exist_ok=True)
            
            # Load data from files
            bands = [np.load(os.path.join(old_folder_path, f'band_{i}.npy'), 'rb') for i in range(8, 17)]

            # Merge data and save to a new file
            for i in range(8):
                combined_matrix = np.dstack([band[:,:,i] for band in bands])
                shutil.copy(os.path.join(old_folder_path, 'human_pixel_masks.npy'), new_folder_path)
                frame_file = os.path.join(new_folder_path, f'frame_{i+1}.npy')
                np.save(frame_file, combined_matrix)

    def distribute_folders(self, data_dir, new_data_dir, multiplier=[0.5, 1, 2, 2, 3, 3, 4, 4]):
        """
        Distribute folders based on data from the mask.
        
        Args:
        - data_dir (str): Path to the source directory
        - new_data_dir (str): Path to the new directory
        - multiplier (list): Multipliers for calculating the new folder distribution
        """
        
        # Initialize lists to store folders based on mask content
        folders = [[] for _ in range(8)]
        
        # Distribute folders among lists
        for folder in os.listdir(data_dir):
            mask = np.load(os.path.join(data_dir, folder, 'human_pixel_masks.npy'))
            num_positive_pixels = np.sum(mask > 0)
            index = min(int(num_positive_pixels / 500), 7)
            folders[index].append(folder)

        # Calculate the number of folders for the new directory
        total_folders = sum(len(folder) for folder in folders)
        num_new_folders = [round(len(folder) / total_folders * 20529 * m) for folder, m in zip(folders, multiplier)]

        # Create the new directory
        os.makedirs(new_data_dir, exist_ok=True)
        
        # Copy data to the new directory
        for i in range(8):
            np.random.shuffle(folders[i])
            folders_to_copy = folders[i][:int(num_new_folders[i])]
            for folder in folders_to_copy:
                new_path = os.path.join(new_data_dir, folder)
                os.makedirs(new_path, exist_ok=True)
                for file_name in ['frame_5.npy','frame_4.npy', 'frame_3.npy', 'human_pixel_masks.npy']:
                    src_file_path = os.path.join(data_dir, folder, file_name)
                    dst_file_path = os.path.join(new_path, file_name)
                    if os.path.exists(src_file_path):
                        shutil.copy2(src_file_path, dst_file_path)