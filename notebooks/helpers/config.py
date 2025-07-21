import os
import re

def set_metadata(task: str):
    """
    Sets metadata for the group.

    Args:
        task (str): Specified task [CLS = classification, DETECT = object detection].
        
    Return:
        group (str): Group name in lowercase and without special characters.
        path (str): Path to the working directory of the group.

    Raises:
        None.
    """

    if task == 'CLS':
        task_dir = 'classification'
    if task == 'DETECT':
        task_dir = 'detection'
    special_chars = {ord('ä'):'ae', ord('ü'):'ue', ord('ö'):'oe', ord('ß'):'ss'}
    group = str(input('Gruppenname: ')).lower().strip().translate(special_chars)
    
    while True:
        if '_' in group or group == '':
            print('Group name cannot contain the underscore (_) character or be empty.')
            group = str(input('Gruppenname: ')).lower().strip().translate(special_chars)
        else:
            break
    
    # Path für den Gruppenordner erstellen
    path = os.path.join('..', 'data', task_dir, group)
    
    # Gruppenordner erstellen, falls er nicht existiert
    os.makedirs(path, exist_ok=True)
    print(f"Gruppenordner erstellt: {path}")

    return group, path

def set_image_data(task: str, group_path: str, device: int = 0):
    """
    Sets metadata for image capturing.

    Args:
        task (str): Specified task [CLS = classification, DETECT = object detection].
        group_path (str): Path to the group's working directory.
        device (int): Specifies the image capturing device. Defaults to 0 (integrated camera).
        
    Return:
        name (str): Name of the class (only for classification). Returns None with task detection.
        num_imgs (int): Number of images to be taken.
        delay (float): Time (in seconds) between two image captures.
        device (int): Image capturing device.

    Raises:
        None.
    """
    special_chars = {ord('ä'):'ae', ord('ü'):'ue', ord('ö'):'oe', ord('ß'):'ss'}

    if task == 'CLS':
        name = str(input('Class: ')+'_').lower().strip().translate(special_chars)
        
        # Unterordner für die Klasse im Gruppenordner erstellen
        class_dir = os.path.join(group_path, name)
        os.makedirs(class_dir, exist_ok=True)
        print(f"Klassenordner erstellt: {class_dir}")

    if task == 'DETECT':
        name = None

    num_imgs = int(input('Number of images: '))
    delay = float(input('Delay: '))
    
    return name, num_imgs, delay, device
