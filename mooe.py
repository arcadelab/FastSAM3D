import os
import shutil
import random

random_seed = 123
random.seed(random_seed)

image_folder = '/home/yiqing/project/lighting_sam_3d/data/Train/images'
label_folder = '/home/yiqing/project/lighting_sam_3d/data/Train/label'

val_image_folder = '/home/yiqing/project/lighting_sam_3d/data/val/train'
val_label_folder = '/home/yiqing/project/lighting_sam_3d/data/val/label'

os.makedirs(val_image_folder, exist_ok=True)
os.makedirs(val_label_folder, exist_ok=True)

all_images = [f for f in os.listdir(image_folder) if f.endswith('.pt')]
num_files = len(all_images)

num_validation = 227

if num_files < num_validation:
    print("Not enough files in the data folder.")
    exit()

selected_images = random.sample(all_images, num_validation)

for image_file in selected_images:
    label_file = 'label' + image_file[len('images'):]
    shutil.move(os.path.join(image_folder, image_file), os.path.join(val_image_folder, image_file))
    shutil.move(os.path.join(label_folder, label_file), os.path.join(val_label_folder, label_file))

print(f"Moved {num_validation} image and label files to the validation folders with seed {random_seed}.")
