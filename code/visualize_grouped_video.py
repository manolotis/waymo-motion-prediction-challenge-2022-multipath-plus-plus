# Makes a video/gif out of several images
import cv2
import os


def sortNumber(val):
    val = val.replace(".png", "").split("_")[-1]

    return val


# MODE = "standard"
# MODE = "simplified_rg"
# MODE = "simplified_rg_no_others"
MODE = "carla"

mode2paths = {
    "standard": {
        "image_folder": "/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/viz/img/",
        "video_folder": "/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/viz/mp4/",
    },
    "simplified_rg": {
        "image_folder": "/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/viz/img_simplified_rg/",
        "video_folder": "/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/viz/mp4_simplified_rg/",
    },
    "simplified_rg_no_others": {
        "image_folder": "/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/viz/img_simplified_rg_no_others/",
        "video_folder": "/home/manolotis/sandbox/temporal-consistency-tests/multipathPP/viz/mp4_simplified_rg_no_others/",
    },
    # "carla": {
    #     "image_folder": "",
    #     "video_folder": "",
    # }
}

image_folder = mode2paths[MODE]["image_folder"]
video_folder = mode2paths[MODE]["video_folder"]

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

    # Loop through all images and write them to the video
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)  # Add the frame to the video

    # Release the VideoWriter object
    video.release()

    print("Video created successfully!", k)
