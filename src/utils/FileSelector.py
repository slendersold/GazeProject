from tqdm.notebook import tqdm
import os
import supervision as sv
import shutil
import glob
from datetime import date, datetime, timezone
import json
import cv2

# Поиск всех видео
mp4_paths = {"path": [], "change": []}


# Функция для поиска путей к файлам
def find_paths_in_subfolders(start_directory, target_folder_name):
    result_paths = []
    for root, dirs, files in os.walk(start_directory):
        if target_folder_name in dirs:
            target_path = os.path.join(root, target_folder_name)
            for file_name in os.listdir(target_path):
                file_path = os.path.join(target_path, file_name)
                result_paths.append(file_path)
    return result_paths


def find_all_proper_videos_by_walk(
    path_to_vid_search, file_format=".mp4", filter_folder_name="prepaired"
):
    for root, dirs, files in os.walk(path_to_vid_search):
        for file_ in files:
            if file_.endswith(file_format) and filter_folder_name in root:
                print(file_)
                mp4_paths["path"].append(os.path.join(root, file_))
                mp4_paths["change"].append(
                    (
                        mp4_paths["path"][-1].split("\\")[-4],
                        mp4_paths["path"][-1].split("\\")[-3],
                    )
                )


def copy_files_with_new_names(destination_path):
    cnt = 0
    for old_path, adder in zip(mp4_paths["path"], mp4_paths["change"]):
        print(f'{cnt} / {len(mp4_paths["path"])}', end="\r")
        # Extracting file name from old path
        old_filename = os.path.basename(old_path)

        # Generating new file name (modify this logic as per your requirement)
        new_filename = (
            adder[0] + "_" + adder[1] + "_" + old_filename
        )  # For example, prefixing "new_" to old filename

        # Constructing new path with destination directory and new filename
        new_path = os.path.join(destination_path, new_filename)

        # Copying file from old path to new path
        shutil.copy(old_path, new_path)
        cnt += 1
    print(f'{cnt} / {len(mp4_paths["path"])}', end="\r")


def save_frames_by_step(
    Video_paths,
    Saving_path,
    FRAME_STRIDE=10,
    video_file_format="mp4",
    picture_file_format=".png",
    clear=False,
):
    # Вытаскиваем фрэймы из видео
    VIDS_PATH = sv.list_files_with_extensions(
        directory=Video_paths, extensions=[video_file_format]
    )

    for video_path in tqdm(VIDS_PATH):
        video_name = video_path.stem
        image_name_pattern = video_name + "-{:05d}" + picture_file_format
        with sv.ImageSink(
            target_dir_path=Saving_path, image_name_pattern=image_name_pattern
        ) as sink:
            for image in sv.get_video_frames_generator(
                source_path=str(video_path), stride=FRAME_STRIDE
            ):
                sink.save_image(image=image)

    if clear:
        files = glob.glob(f"{Video_paths}\\*")
        for f in files:
            os.remove(f)
