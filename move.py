import os
import shutil
import random

random_seed = 123
random.seed(random_seed)

image_folder = 'data/initial_train_dataset/images'
label_folder = 'data/initial_train_dataset/label'

val_image_folder = 'data/validation/edema/mr_flair_BraTS2021/imagesTr'
val_label_folder = 'data/validation/edema/mr_flair_BraTS2021/labelsTr'

all_images = [f for f in os.listdir(image_folder) if f.endswith('.nii.gz')]
num_files = len(all_images)

num_validation = 170

if num_files < num_validation:
    print("Not enough files in the data folder.")
    exit()

selected_images = random.sample(all_images, num_validation)

for image_file in selected_images:
    label_file = image_file
    shutil.move(os.path.join(image_folder, image_file), os.path.join(val_image_folder, image_file))
    shutil.move(os.path.join(label_folder, label_file), os.path.join(val_label_folder, label_file))

print(f"Moved {num_validation} image and label files to the validation folders with seed {random_seed}.")
