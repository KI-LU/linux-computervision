import os
from ultralytics import YOLO
import cv2
from pprint import pprint
import numpy as np
import re

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

def get_trained_model(path: str, run=None) -> str:
    """
    Select the subdirectory with the highest number from a given directory and accesses the trained model If a run is specified, returns model of that run.

    Args:
        path (str): The directory path containing the numbered subdirectories.
        run (int): Specific experiment run. Default: None.

    Returns:
        model (str): Path to trained model.
    """

    group = path.split('\\')[-1] if os.name == 'nt' else path.split('/')[-1] # extract group name depending on operating system
    path = os.path.join(path, 'runs')
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    highest_number = -1
    highest_subdir = ""

    # if run is specified, get selected run
    # todo: error handling -> not for me, for you. When there are errors. inevitably. have fun!
    if run is not None and run != 1:
        specific_run = str(run)
        for subdir in subdirs:
            if subdir.endswith(specific_run):
                target_subdir = subdir
                break
        model = os.path.join(path, target_subdir, 'weights', 'best.pt')
        print(f'Using {model}')
        return model

    # if run is 1 or not specified, get latest model
    for subdir in subdirs:
        match = re.search(r'(\d+)$', subdir)
        if match:
            number = int(match.group())
            if number > highest_number:
                highest_number = number
                highest_subdir = subdir

    if highest_subdir == "" or run == 1:
        highest_subdir = group

    latest_model = os.path.join(path, highest_subdir, 'weights', 'best.pt')
    print(f'Using {latest_model}')
    return latest_model


def inference_video(video_path: str,
                    model,
                    verbose=True) -> None:
    """
    Performs inference on a video file.

    Args:
        video_path (str): Path to the video file.
        model (.pt): Instance of YOLO-model that performs inference.
        verbose (bool): Flag to decide whether model output is shown.

    Return:
        None.

    Raises:
        None.
    """
    # defining capturing device (in this case: path)
    cap = cv2.VideoCapture(video_path) 

    # looping over the video frames and performing inference on each frame
    while cap.isOpened():
        success, frame = cap.read() # reading frame
    
        if success: # if frame was successfully read
    
            # inference
            results = model(frame, verbose=verbose)
    
            result_image = results[0].plot()
            # Convert the result to a format suitable for OpenCV if needed
            if isinstance(result_image, np.ndarray):
                cv2.imshow("Model Prediction", result_image) # display frame
    
            # break out of loop by pressing q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # break out of loop once end of video file is reached
            break
    
    # clean up: close all windows
    cap.release()
    cv2.destroyAllWindows()


def inference_webcam(model,
                     device: int = 0,
                     verbose=False) -> None:
    """
    Performs inference on a video file.

    Args:
        model (.pt): Instance of YOLO-model that performs inference.
        device (int): Specifies capturing device.
        verbose (bool): Flag to decide whether model output is shown. Defaults to False.

    Return:
        None.

    Raises:
        None.
    """
    # accessing the capturing device
    cap = find_device_port(device)
    print('Camera recognized: ', cap.open(device))

    # looping over all frames captured
    while cap.isOpened():
        success, frame = cap.read() # reading frame
    
        if success: # if frame was successfully read
    
            # inference
            results = model(frame, verbose=verbose)
    
            result_image = results[0].plot()
            # Convert the result to a format suitable for OpenCV if needed
            if isinstance(result_image, np.ndarray):
                cv2.imshow("Model Prediction", result_image) # display frame
    
            # break out of loop by pressing q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # break out of loop once end of video file is reached
            break
    
    # clean up: close all windows
    cap.release()
    cv2.destroyAllWindows()


def choose_model(task: str):
    """
    Performs inference on a video file.

    Args:
        task (str): Defining the task [CLS = classification, DETECT = object detection].
        
    Return:
        model (.pt): Instacne of YOLO-model object.

    Raises:
        None.
    """

    # Defining list of possible tasks --> extendable
    task_list = [
        'CLS',
        'DETECT'
    ]

    # defining available models for classification
    cls_model_dict = {
        1: 'yolov8n-cls (Nano)',
        2: 'yolov8s-cls (Small)',
        3: 'yolov8m-cls (Medium)',
        4: 'yolov8l-cls (Large)',
        5: 'yolov8x-cls (Very large)'
    }

    # defining available models for detection
    detect_model_dict = {
        1: 'yolov8n (Nano)',
        2: 'yolov8s (Small)',
        3: 'yolov8m (Medium)',
        4: 'yolov8l (Large)',
        5: 'yolov8x (Very large)'
    }

    # displaying available models and taking in user choice, then returning model
    print('The available models for this task are:\n')
    if task == 'CLS':
        pprint(cls_model_dict)
        choice = int(input('Choose a model by entering the model number: '))
        model = cls_model_dict.get(choice)
        return YOLO(os.path.join('..', 'models', model.split()[0]+'.pt'))
        
    # displaying available models and taking in user choice, then returning model
    if task == 'DETECT':
        pprint(detect_model_dict)
        choice = int(input('Choose a model by entering the model number: '))
        model = detect_model_dict.get(choice)
        return YOLO(os.path.join('..', 'models', model.split()[0]+'.pt'))

    if task not in task_list:
        print('Please choose a valid task.')
        return None


def set_training_config(path: str):
    """
    Sets configuration for model training.

    Args:
        path (str): Path to data directory.
        
    Return:
        epochs (.pt): Instacne of YOLO-model object.
        data_path (str): Path to config.yaml file
        name (str): Experiment name.
        save_dir (str): Path to save directory.

    Raises:
        None.
    """
    epochs = int(input('How many epochs: '))
    data_path = os.path.join(path, 'config.yaml')
    group = path.split('\\')[-1] if os.name == 'nt' else path.split('/')[-1] # extract group name depending on operating system
    name = f'{group}'
    save_dir = os.path.join(path,'runs')
    return epochs, data_path, name, save_dir
                    




















