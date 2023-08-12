class IndividualDataLoader:

    def __init__(self, directory, target_size=(256,256), augment=True, visible_light = True):
        self.directory = directory
        self.target_size = target_size
        self.augment = augment
        self.visible_light = visible_light
        self.folders = os.listdir(self.directory)
        if self.augment:
            self.augmentor = ImageDataGenerator(
                rotation_range=90,       
                width_shift_range=0.05,   
                height_shift_range=0.05,  
                zoom_range=0.05,          
                horizontal_flip=True,    
                vertical_flip=True,      
                fill_mode='nearest'      
            )
            
    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        folder = self.folders[index]
        X = []
        y = []

        _T11_BOUNDS = (243, 303)
        _CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
        _TDIFF_BOUNDS = (-4, 2)
        
        seed = np.random.randint(0, 10000)
        frame5 = np.load(os.path.join(self.directory, folder, "frame_5.npy"))
        mask = np.load(os.path.join(self.directory, folder, "human_pixel_masks.npy"))
        
        frame5 = resize(frame5, self.target_size, mode='reflect', preserve_range=True)
        mask = resize(mask, self.target_size, mode='reflect', preserve_range=True)
        
        frame_norm5 = (frame5 - 135) / (340 - 135)
        if self.visible_light:
            r = self.normalize_range(frame5[:,:,7] - frame5[:,:,6], _TDIFF_BOUNDS)
            g = self.normalize_range(frame5[:,:,6] - frame5[:,:,3], _CLOUD_TOP_TDIFF_BOUNDS)
            b = self.normalize_range(frame5[:,:,6], _T11_BOUNDS)
            false_color5 = np.clip(np.stack([r, g, b], axis=2), 0, 1)
            false_color5 = resize(false_color5, self.target_size, mode='reflect', preserve_range=True)
            frame5 = np.concatenate((frame_norm5, false_color5 ), axis=2)



        frame4 = np.load(os.path.join(self.directory, folder, "frame_4.npy"))
        frame4 = resize(frame4, self.target_size, mode='reflect', preserve_range=True)
        
        frame_norm4 = (frame4 - 135) / (340 - 135)
        if self.visible_light:
            r = self.normalize_range(frame4[:,:,7] - frame4[:,:,6], _TDIFF_BOUNDS)
            g = self.normalize_range(frame4[:,:,6] - frame4[:,:,3], _CLOUD_TOP_TDIFF_BOUNDS)
            b = self.normalize_range(frame4[:,:,6], _T11_BOUNDS)
            false_color4 = np.clip(np.stack([r, g, b], axis=2), 0, 1)
            false_color4 = resize(false_color4, self.target_size, mode='reflect', preserve_range=True)
            frame4 = np.concatenate((frame_norm4, false_color4), axis=2)


        frame3 = np.load(os.path.join(self.directory, folder, "frame_3.npy"))            
        frame3 = resize(frame3, self.target_size, mode='reflect', preserve_range=True)
        
        frame_norm3 = (frame3 - 135) / (340 - 135)
        if self.visible_light:
            r = self.normalize_range(frame3[:,:,7] - frame3[:,:,6], _TDIFF_BOUNDS)
            g = self.normalize_range(frame3[:,:,6] - frame3[:,:,3], _CLOUD_TOP_TDIFF_BOUNDS)
            b = self.normalize_range(frame3[:,:,6], _T11_BOUNDS)
            false_color3 = np.clip(np.stack([r, g, b], axis=2), 0, 1)
            false_color3 = resize(false_color3, self.target_size, mode='reflect', preserve_range=True)
            frame3 = np.concatenate((frame_norm3, false_color3), axis=2)

        frame = np.concatenate((frame3, frame4, frame5), axis=2)

        if self.augment:
            frame = self.augmentor.random_transform(frame, seed=seed)
            mask = self.augmentor.random_transform(mask, seed=seed)

        X.append(frame)
        y.append(mask)

        return np.array(X), np.array(y)

    def normalize_range(self, data, bounds):
        """Maps data to the range [0, 1]."""
        return (data - bounds[0]) / (bounds[1] - bounds[0])