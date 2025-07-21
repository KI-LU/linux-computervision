import os
import time
import shutil
from pathlib import Path
from datetime import datetime, timedelta

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Ordner konfigurieren
SOURCE_DIR = Path("/home/lernumgebung/Pictures/Webcam")
DETECTION_DIR = Path("/home/lernumgebung/Desktop/modules/computervision/data/detection")

# Bild-Endungen
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']

# Funktion, um den neuesten Unterordner zu finden
def get_latest_subfolder(base_path: Path) -> Path | None:
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    return max(subdirs, key=lambda d: d.stat().st_mtime)

# Funktion zum Verschieben von Bildern, die in den letzten 8 Stunden erstellt wurden
def move_recent_images(source_dir: Path, target_base_dir: Path, hours: int = 8):
    now = time.time()
    cutoff = now - (hours * 3600)  # Berechne die Zeit vor 8 Stunden

    files_moved = 0
    latest_target_folder = get_latest_subfolder(target_base_dir)
    if not latest_target_folder:
        print("⚠️ Kein Unterordner in detection gefunden.")
        return

    # Durchsuche alle Dateien im Quellordner
    for file in source_dir.iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            created_time = file.stat().st_ctime  # Datei-Erstellzeit
            if created_time >= cutoff:
                try:
                    shutil.move(str(file), latest_target_folder / file.name)
                    print(f"✅ {file.name} verschoben nach {latest_target_folder}")
                    files_moved += 1
                except Exception as e:
                    print(f"❌ Fehler beim Verschieben {file.name}: {e}")

    if files_moved == 0:
        print("ℹ️ Keine neuen Bilder gefunden.")
    else:
        print(f"📦 Insgesamt verschoben: {files_moved} Bild(er)")

# Event-Handler, der nur bei neuen Dateien aufgerufen wird
class CheeseImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        print(f"[DEBUG] Event erkannt: {event.src_path}")

        if event.is_directory:
            return

        filepath = Path(event.src_path)
        print(f"[DEBUG] Datei erstellt: {filepath.name}")

        if filepath.suffix.lower() in IMAGE_EXTENSIONS:
            print(f"📷 Neues Bild erkannt: {filepath.name}")
            target_folder = get_latest_subfolder(DETECTION_DIR)

            if target_folder:
                try:
                    shutil.move(str(filepath), target_folder / filepath.name)
                    print(f"✅ {filepath.name} verschoben nach {target_folder}")
                except Exception as e:
                    print(f"❌ Fehler beim Verschieben: {e}")
            else:
                print("⚠️ Kein Zielordner in 'detection/' gefunden!")
        else:
            print(f"[DEBUG] Nicht unterstützte Endung: {filepath.suffix}")

# Hauptteil: Beobachte den Quellordner und verschiebe Bilder, die in den letzten 8 Stunden erstellt wurden
if __name__ == "__main__":
    print(f"🔍 Beobachte Ordner: {SOURCE_DIR}")

    # Beobachten des Ordners und Verschieben von Bildern, die in den letzten 8 Stunden erstellt wurden
    move_recent_images(SOURCE_DIR, DETECTION_DIR, hours=8)

    event_handler = CheeseImageHandler()
    observer = Observer()
    observer.schedule(event_handler, path=str(SOURCE_DIR), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 Beendet.")
        observer.stop()
    observer.join()
