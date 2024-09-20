# Makes a video/gif out of several images

import cv2
import os
import pdb


def sortNumber(val):
    val = val.replace(".png","").split("_")[-1]

    return val


# Specify the directory containing your PNG images
image_folder = '/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/viz/img/'
video_folder = '/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/viz/mp4/'
gif_folder = '/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/viz/gif/'

# Get a list of image filenames in the folder, sorted by filename
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

mapping = {}

# Group per
for file in images:
    f = file.split("/")[-1].replace(".npz", "")
    scid, aid, atype, t = f.split("__")
    substring_filter = f"{scid}__{aid}"
    if substring_filter not in mapping:
        mapping[substring_filter] = []
    mapping[substring_filter].append(file)

for k, images in mapping.items():

    images.sort(key=sortNumber)  # Sort files to maintain the correct order
    # Read the first image to get the size
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_image.shape

    # Define the codec and create a VideoWriter object
    video = cv2.VideoWriter(video_folder + k + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
    # gif = cv2.VideoWriter(video_folder + k + '.gif', cv2.VideoWriter_fourcc(*'gif'), 10, (width, height))

    # Loop through all images and write them to the video
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)  # Add the frame to the video
        # gif.write(frame)  # Add the frame to the video

    # Release the VideoWriter object
    video.release()
    # gif.release()

    print("Video created successfully!", k)
