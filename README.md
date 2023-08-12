# Kaggle_google_segmentation

[Русский](README_RUS.md)

## Identification of Contrails Based on Satellite Imagery
### Kaggle Competition: Google Research: Identify Contrails to Reduce Global Warming

### Project Objective
Develop a model that, by analyzing satellite imagery, will help improve the prediction of contrail formation from aircraft engines. These contrails contribute to global warming, making their monitoring critical for the environment.

### My Reflection on the Competition
This competition marked my debut on the Kaggle platform. Although my initial outcome wasn't as expected, every phase of the competition provided invaluable experience.

After examining the winners' solutions post-competition, I identified several key takeaways:

- Increasing image resolution: The top performers increased the resolution from 256 to 512 and even to 1024, significantly influencing prediction quality. This practice was new to me, and it gave me a profound understanding of its importance.
- Data augmentation errors: I overlooked some augmentation nuances that could have affected the final outcome (explanation in the comment).
- Modern architectures: Most winners employed UNet models with transformers. Delving into these solutions, I realized I missed out on implementing the latest deep learning advancements.
Despite the above points, I noticed that in most aspects, my actions aligned with the winners'. This reaffirms that the core direction of my research and development was on track.

### Data
Geostationary satellite imagery from GOES-16 ABI is used to identify aircraft contrails. The data consists of a sequence of images taken every 10 minutes, where each example contains exactly one labeled picture.

#### Data Structure:
- train/: Training set.

- band_{08-16}.npy: Image sequences for different infrared channels.

- human_individual_masks.npy: Masks labeled by individual experts.

- human_pixel_masks.npy: Aggregated masks based on majority expert opinions.

- validation/: Validation set. Contains the same files as train/ but without human_individual_masks.npy.

- test/: Testing set. Contains images that need to be forecasted.

- {train|validation}_metadata.json: Metadata for each entry, including timestamps and projection parameters.

### Tools and Libraries
Python
Keras: for constructing and training neural network models.
Numpy: for data array handling.
TensorFlow: as the primary library for deep learning.
Model - R2U-Net