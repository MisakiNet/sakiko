import os
import shutil

import cv2

# Define the classification categories
categories = ['nokey', 'miss', 'bad', 'good', 'great', 'perfect', 'z-done']

target_dir = 'test'

# Create the train directories if they don't exist
for category in categories:
    dir_path = f'{target_dir}/{category}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Load all PNG files in the temp directory
image_files = [f for f in os.listdir('temp') if f.endswith('.png')]

# Display each image and classify it
for file in image_files:
    img = cv2.imread(f'temp/{file}')
    cv2.imshow('Image', img)
    key = cv2.waitKey(0) & 0xFF

    # Classify the image based on keyboard input
    if key == ord('n'):
        category = categories[0]
    elif key == ord('m'):
        category = categories[1]
    elif key == ord('b'):
        category = categories[2]
    elif key == ord('d'):
        category = categories[3]
    elif key == ord('t'):
        category = categories[4]
    elif key == ord('p'):
        category = categories[5]
    elif key == ord('z'):
        category = categories[6]
    else:
        print("Invalid input. Skipping image.")
        continue

    # Move the classified image to the corresponding directory
    shutil.move(f'temp/{file}', f'{target_dir}/{category}/{file}')
    print(f"Image {file} classified as {category}")

cv2.destroyAllWindows()
