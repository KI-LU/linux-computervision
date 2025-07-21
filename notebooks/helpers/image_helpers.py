import cv2, time, os
import math
import matplotlib.pyplot as plt
from matplotlib.image import imread

def find_device_port(device):
    """Finds correct device port and connects to it."""
    print(f'Trying device {device}')
    cap = cv2.VideoCapture(device)
    device = 0
    while not cap.isOpened() and device <= 200:
        print(f'Trying device {device}')
        cap = cv2.VideoCapture(device)
        if cap.isOpened():
            break
        device += 1
    if not cap.isOpened():
        print('Connection to camera could not be established.')
    print(f'Successful with device port: {device}')
    return cap

def capture_images(num_imgs: int,
                   name: str,
                   img_path: str,
                   device: int = 0,
                   delay: float = 3.0,
                   show: bool = False) -> None:
    """
    Captures images from a specified video device and saves them to disk as .png.

    Args:
        num_imgs (int): The number of images to capture per class.
        name (str): A string for identifying the images.
        img_path (str): The directory path where the captured images will be saved.
        device (int, optional): The index of the video capturing device. Defaults to 0.
        delay (float, optional): The delay in seconds between capturing each image. Defaults to 3.
        show (bool, optional): If True, displays the current captured image during the process. Defaults to False.

    Returns:
        None

    Raises:
        None
    """

    # accesses the camera
    cap = find_device_port(device)

    # creates a directory (if not already existent) in which to save the images
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    # enumerates the images already present in the directory and prints the number
    # this way we start numbering them from the largest number already present in the directory
    # e.g. 10 images already present, this time the numbering starts from 10 --> img_10, img_11, ...
    files = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))]
    i = len(files)
    print(i)
    
    print('Image: {}'.format(name))

    print('Camera recognized: ', cap.open(device))

    # looping over all frames captured
    for img_num in range(num_imgs):
        if cv2.waitKey(1) & 0x20 == ord(" "):
            print('Capturing {}, Image {}'.format(name, i+1))
            
            ret, frame = cap.read() # read frame
            imgname = os.path.join(img_path, name + '_' + str(i) + '.png') # create image name
            i += 1 # increment counter variable
            cv2.imwrite(imgname, frame) # save image to directory

        if show:
            cv2.imshow('Current image', frame)

        time.sleep(delay)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # window wont open

        
    # clean-up: close all windows and display the directory path under which the images are saved       
    cap.release()
    cv2.destroyAllWindows()
    print(f'Images saved at {img_path}')


# FUNCTION NOT USED CURRENTLY --> MAY REMOVE IN THE FUTURE
def display_images(directory_path: str) -> None:
    """
    Display all images in the specified directory in a grid using Matplotlib.

    Args:
        directory_path (str): The path to the directory containing the images.

    Returns:
        None
    """

    # counts displayable images in the directory and puts them in a list
    files = os.listdir(directory_path)
    image_files = [file for file in files if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # check if enough images are in the directory
    num_images = len(image_files)
    if num_images < 5:
        print('Please add more images. Minimum amount is 5.')
        return

    # define col and row amount (hardcoded cols --> not good, this function could be reworked in the future)
    cols = 4  
    rows = math.ceil(num_images / cols)

    # creating figure
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))

    # plot images in figure
    for i, file in enumerate(image_files):
        row = i // cols
        col = i % cols
        img_path = os.path.join(directory_path, file)
        img = imread(img_path)
        if img is not None:
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            axes[row, col].set_title(file.split('.')[0])

    # remove superflouos axes
    for ax in axes.flat:
        if not ax.images:
            ax.remove()

    # display
    plt.tight_layout()
    plt.show()


