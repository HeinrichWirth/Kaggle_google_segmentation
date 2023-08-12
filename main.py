import tensorflow as tf
from tensorflow import keras
from keras_unet_collection import models
import segmentation_models as sm
import tensorflow as tf
from keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback
from skimage.transform import resize
from tensorflow.keras import backend as K
from segmentation_models.metrics import FScore
from keras_unet_collection import models
from DataProcessor import DataProcessor
from DataLoader import IndividualDataLoader
import configparser

#Loss function
def weighted_binary_crossentropy_factory(bce):
    def weighted_binary_crossentropy(y_true, y_pred):

        if bce < 0.7:
            coeff = 3
        elif bce < 0.9:
            coeff = 1
        elif bce < 0.95:
            coeff = 1
        else:
            coeff = 1

        class_weight_for_0 = 1 - bce
        weight_vector = y_true * bce + (1. - y_true) * class_weight_for_0
        loss = K.binary_crossentropy(y_true, y_pred)
        weighted_loss = loss * weight_vector
        return coeff * K.mean(weighted_loss)
    return weighted_binary_crossentropy

def weighted_dice_loss_factory(weight):
    def weighted_dice_loss(y_true, y_pred):
        smooth = 0.0001
        w = weight * y_true
        intersection = K.abs(tf.reduce_sum(w * y_pred, axis=[1,2,3]))
        score = (2. * intersection + smooth) / (tf.reduce_sum(w, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3]) + smooth)
        loss = 1. - K.mean(score)
        return loss
    return weighted_dice_loss

def custom_loss(y_true, y_pred):
    return  weighted_dice_loss(y_true, y_pred) + weighted_binary_crossentropy(y_true, y_pred)

def write_logs_to_file(epoch, logs, path, BACKBONE):
    with open(f'{path}{BACKBONE}_training_logs.txt', 'a') as file:
        file.write(f"Эпоха: {epoch}, Логи: {logs}\n")


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')

    #Files preprocessing
    #Train
    processor = DataProcessor()
    train_dir = config.get('PATHS', 'train_dir')
    new_train_dir = config.get('PATHS', 'new_train_dir')
    processor.process_directory(train_dir, new_train_dir)

    data_dir_train = config.get('PATHS', 'data_dir_train')
    new_data_dir_train = config.get('PATHS', 'new_data_dir_train')
    processor.distribute_folders(data_dir_train, new_data_dir_train)

    #Val
    val_dir = config.get('PATHS', 'val_dir')
    new_val_dir = config.get('PATHS', 'new_val_dir')
    processor.process_directory(val_dir, new_val_dir)

    data_dir_val = config.get('PATHS', 'data_dir_val')
    new_data_dir_val = config.get('PATHS', 'new_data_dir_val')
    processor.distribute_folders(data_dir_val, new_data_dir_val)

    #DataLoader creating
    train_generator = IndividualDataLoader(new_data_dir_train, batch_size=4, shuffle=True, visible_light = False)
    val_generator = IndividualDataLoader(new_data_dir_val, batch_size=4, shuffle=False, augment=False, visible_light= False)
    
    f_score = FScore()

    mod = 'r2_unet_2d_gelu'

    model_code = {
                'r2_unet_2d_gelu' : models.r2_unet_2d(input_size=(256, 256, 9), filter_num=[64, 128, 256, 512], n_labels=1,
                            stack_num_down=2, stack_num_up=1, recur_num=2,
                            activation='GELU', output_activation='Sigmoid', 
                            batch_norm=True, pool='ave', unpool='bilinear', name='r2unet')
                }

    loss_func = {'r2_unet_2d_gelu':'val_iou_score'}
    weighted_dice_loss = weighted_dice_loss_factory(1)
    weighted_binary_crossentropy = weighted_binary_crossentropy_factory(0.5)
    BACKBONE = mod
    model = model_code[mod]

    # Set the path to the folder for saving the model
    model_save_folder = config.get('[MODEL]', 'model_save_folder')

    # Set the file name for saving the model
    model_save_name = config.get('[MODEL]', 'model_save_name')
    model_save_name = model_save_folder.format(model_save_folder = model_save_name, mod = mod)

    # Create a ModelCheckpoint object
    checkpoint = ModelCheckpoint(model_save_name, monitor=loss_func[mod], verbose=1, save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor=loss_func[mod], factor=0.9, patience=5, verbose=1, min_lr=0.00001)
    log_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: write_logs_to_file(epoch, logs, model_save_folder, BACKBONE))

    # Set the initial learning rate value
    initial_learning_rate = 0.0001

    optimizer = Adam(learning_rate=initial_learning_rate)
    model.compile(
    optimizer,
    loss=custom_loss,
    metrics=[sm.metrics.iou_score,f_score, weighted_dice_loss, weighted_binary_crossentropy],
    )

    history = model.fit(
    train_generator,
    batch_size=4,
    epochs=30,
    validation_data=val_generator,
    callbacks=[checkpoint, reduce_lr, log_callback]
    )