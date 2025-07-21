from sklearn.model_selection import train_test_split
import os
import shutil
import yaml

def show_directory(path: str, 
                   show_files=False) -> None:
    import os
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        if (os.path.basename(root).startswith('.') or os.path.basename(root).startswith('__')):
            continue
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        if show_files:
            for f in files:
                print(f"{subindent}{f}")


def move_file(source_path: str,
              destination_path: str) -> None:
    """
    Copy a file from source_path to destination_path, prepending a unique ID to the filename.

    Args:
        source_path (str): The path to the source file.
        destination_path (str): The path to the destination directory where the file will be copied.

    Returns:
        None
    """
    import os, shutil, uuid

    try:
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"The file '{source_path}' does not exist.")

        unique_id = str(uuid.uuid4())
        filename, extension = os.path.splitext(os.path.basename(source_path))
        new_filename = f"{unique_id}_{filename}{extension}"
        destination_path_with_id = os.path.join(os.path.dirname(destination_path), new_filename)
        os.makedirs(os.path.dirname(destination_path_with_id), exist_ok=True)
        shutil.copy2(source_path, destination_path_with_id)

        print(f"File '{source_path}' successfully copied to '{destination_path_with_id}'.")
    except Exception as e:
        print(f"An error occurred: {e}")


def prepare_folder_structure(path: str,
                             task: str,
                             val_size: float = None,
                             test_size: float = None):
    """
    Prepare the folder structure for a dataset depending on the task.
    Args:
        path (str): The directory path containing the images.
        task (str): Either 'CLS' for classification or 'DETECT' for object detection.
        val_size (float): The proportion of the dataset to include in the validation split from the initial split.
        test_size (float): The proportion of the dataset to include in the test split from the remaining data after the validation split.
    Returns:
        Function depending on task.
    """

    def prepare_detect_folder_structure(path: str, val_size: float):
        """
        Prepare the folder structure for an object detection dataset by splitting images and labels into training and validation sets.
        Args:
            path (str): The directory path containing the 'images' and 'labels' subdirectories.
            val_size (float): The proportion of the dataset to include in the validation split.
        Raises:
            ValueError: If no valid image-label pairs are found.
        Returns:
            None
        """
        img_path = os.path.join(path, 'images')
        label_path = os.path.join(path, 'labels')

        # Sammle alle Bild-Label-Paare robust
        valid_img, valid_label = [], []
        for img in os.listdir(img_path):
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                base = os.path.splitext(img)[0]
                lbl_file = base + '.txt'
                if os.path.exists(os.path.join(label_path, lbl_file)):
                    valid_img.append(img)
                    valid_label.append(lbl_file)

        if not valid_img:
            raise ValueError(f"Keine Bilder gefunden in {img_path}!")

        # Split in Training und Validation
        x_train, x_val, y_train, y_val = train_test_split(
            valid_img,
            valid_label,
            test_size=val_size,
            train_size=1 - val_size,
            random_state=42,
            shuffle=True
        )

        # Ordnerstruktur anlegen
        train_img_dest = os.path.join(img_path, 'train')
        val_img_dest   = os.path.join(img_path, 'val')
        train_lbl_dest = os.path.join(label_path, 'train')
        val_lbl_dest   = os.path.join(label_path, 'val')
        os.makedirs(train_img_dest, exist_ok=True)
        os.makedirs(val_img_dest,   exist_ok=True)
        os.makedirs(train_lbl_dest, exist_ok=True)
        os.makedirs(val_lbl_dest,   exist_ok=True)

        # Verschiebe Dateien
        for img in x_train:
            shutil.move(os.path.join(img_path, img), train_img_dest)
        for lbl in y_train:
            shutil.move(os.path.join(label_path, lbl), train_lbl_dest)
        for img in x_val:
            shutil.move(os.path.join(img_path, img), val_img_dest)
        for lbl in y_val:
            shutil.move(os.path.join(label_path, lbl), val_lbl_dest)

    def prepare_cls_folder_structure(path: str,
                                     val_size: float,
                                     test_size: float):                   
        """
        Prepare the folder structure for a classification dataset by splitting images into training, validation, and test sets.
        """
        classes = dict()
        for img in os.listdir(path):
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                prefix = img.split('_')[0]
                classes.setdefault(prefix, []).append(img)
        split = dict()
        for key, imgs in classes.items():
            train, tmp = train_test_split(imgs, test_size=val_size)
            val, test = train_test_split(tmp, test_size=test_size)
            split[key] = (train, val, test)
        for idx, subset in enumerate(['train', 'val', 'test']):
            for key, (train, val, test) in split.items():
                cls_dest = os.path.join(path, subset, key)
                os.makedirs(cls_dest, exist_ok=True)
                for img in split[key][idx]:
                    shutil.move(os.path.join(path, img), os.path.join(cls_dest, img))

    if task == 'CLS':
        val_size = 0.4 if val_size is None else val_size
        test_size = 0.1 if test_size is None else test_size
        return prepare_cls_folder_structure(path, val_size, test_size)
    if task == 'DETECT':
        val_size = 0.2 if val_size is None else val_size
        return prepare_detect_folder_structure(path, val_size)


def create_config(path: str, task: str):
    """
    Generate a configuration file (`config.yaml`) for a specified machine learning task.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'The specified path does not exist')

    if task == 'DETECT':
        def read_class_names(classes_path):
            with open(classes_path, 'r') as file:
                return [line.strip() for line in file]
        classes = read_class_names(os.path.join(path, 'classes.txt'))
        data = {
            'train': 'images/train',
            'val':   'images/val',
            'nc':    len(classes),
            'names': {i: name for i, name in enumerate(classes)}
        }
    if task == 'CLS':
        subdirs = sorted(os.listdir(os.path.join(path, 'train')))
        data = {
            'train': 'train',
            'val':   'val',
            'test':  'test',
            'nc':    len(subdirs),
            'names': subdirs
        }
    with open(os.path.join(path, 'config.yaml'), 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    print(f'Created config.yaml under {path}')


def cleanup_images(path: str) -> None:
    for file in os.listdir(path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            os.remove(os.path.join(path, file))
