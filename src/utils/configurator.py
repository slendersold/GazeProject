import os
import json

# Массив для названий полей
path_fields = (
    "MAIN_PATH_FOR_FILES",
    "PATH_TO_INDEX",
    "PATH_TO_RESULTS",
    "PATH_TO_TRUEMASKS",
    "PATH_TO_EXPMASKS",
    "WEIGHT_PATH",
    "gaze_data_path",
    "input_path",
    "output_path",
    "output_dataframe",
    "output_dataframe_all",
    "sam2_checkpoint",
    "sam_model_weight"
)
filename_fields = (
    "input_video",
    "output_video",
    "output_dataframe",
    "output_dataframe_all",
    "gaze_data_file"
)
variable_fields = (
    "gaze_data_keys",
    "segmentation_classes"
)
data_types = ("str", "int", "list", "dict")
# Библиотека с полями и текстом для запроса у пользователя
instructions = {
    "pth_sgn": ("str", "Введите значение для pth_sgn (По умолчанию это '\\'): ", "\\"),
    "PATH_TO_INPUT": ("str", "Введите путь папки, где будет лежать видеофайл и данные взгляда ", None),
    "PATH_TO_RESULTS": ("str", "Введите путь папки, куда будут записываться результаты: ", None),
    "PATH_TO_TRUEMASKS": ("str", "Введите путь папки, где будут лежать истинные маски ", ""),
    "PATH_TO_EXPMASKS": ("str", "Введите путь папки, куда будут сохраняться экспериментальные бинарные маски ", ""),
    "gaze_data_keys": ("list", 
                       "Введите ключи для данных взгляда через запятую (по умолчанию: ['raw']): ",
                       ["raw"]),
    "gaze_data_file": ("dict", 
                       f"У поля gaze_data введите название файла для ключа ", 
                       {"key_list":"gaze_data_keys", "default_dict": {"raw": "gaze_positions.csv"}}),
    "input_video": ("str", 
                    "Введите название входного видео (По умолчанию это world.mp4) ", 
                    "world.mp4"),
    "output_video": ("str", 
                     "Введите название выходного видео (По умолчанию segmented_video.mp4) ", 
                     "segmented_video.mp4"),
    "output_dataframe": ("str", 
                         "Введите название выходного файла сбора статистики (По умолчанию output_dataframe.csv) ", 
                         "output_dataframe.csv"),
    "output_dataframe_all": ("str", 
                             "Введите название выходного файла сбора всей статистики (По умолчанию output_dataframe_all.csv) ", 
                             "output_dataframe_all.csv"),
    "segmentation_classes": ("list", 
                             "Введите классы сегментации через запятую (По умолчанию: ['base', 'object', 'prosthesis', 'target']): ",
                             ["base","object","prosthesis","target"]),
    "Yolo_model_wheight": ("str", "введите путь до весов модели Yolo, включая название файла ", ""),
    "sam2_checkpoint": ("str", "Введите путь до чекпоинта sam2, включая название файла ", ""),
    "sam_model_weight": ("str", "Введите путь до весов sam2, включая название файла ", "")
}
fields = {}

def check_file_exists(path):
    """Проверяет, существует ли файл по указанному пути."""
    return os.path.exists(path)

def create_or_update_config():
    """Создает или обновляет файл конфигурации, запрашивая значения у пользователя."""
    global fields
    fields = {}

    # Спрашиваем у пользователя, куда сохранить файл конфигурации
    print("Введите путь для сохранения файла конфигурации (включая имя файла): ", flush=True)
    config_path = input()
    print(f"Настройки будут сохранены в {config_path}", flush=True)

    def get_valid_input(prompt, data_type, default=None):
        while True:
            match data_type:
                case "dict":
                    value = {}
                    # Проверка наличия ключей для словаря и значений по умолчанию
                    key_list = fields[default["key_list"]]
                    default_dict = default["default_dict"]

                    for key in key_list:
                        print(f"{prompt}{key}", flush=True)
                        user_input = input().strip()
                        if user_input:
                            value[key] = user_input
                        else:
                            # Используем значение по умолчанию, если оно задано для ключа
                            value[key] = default_dict.get(key, "")
                            if not value[key]:
                                print(f"Похоже на ключ {key} нет значения по умолчанию, пожалуйста, введите значение.", flush=True)
                                while True:
                                    user_input = input().strip()
                                    if user_input:
                                        value[key] = user_input
                                        break
                                    else:
                                        print("Это поле не может быть пустым. Пожалуйста, введите значение.", flush=True)
            
                case "list":
                    print(prompt)
                    user_input = input().strip()
                    if user_input:
                        value = user_input.split(",")  # Преобразуем в список
                    else:
                        value = default if isinstance(default, list) else list(default)
                case "int":
                    print(prompt)
                    user_input = input().strip()
                    if user_input:
                        try:
                            value = int(user_input)
                        except ValueError:
                            print("Неправильный ввод, ожидалось целое число.", flush=True)
                            continue
                    else:
                        value = default
                case _:
                    print(prompt)
                    user_input = input().strip()
                    value = user_input if user_input else default

            # Проверяем, что значение не пустое, если это необходимо
            if value is not None:
                print(f"Вы ввели это: {value}", flush=True)
                return value
            else:
                print("Это поле не может быть пустым. Пожалуйста, введите значение.", flush=True)



    # Заполняем каждое поле на основе instructions
    for key, (data_type, prompt, default) in instructions.items():
        fields[key] = get_valid_input(prompt, data_type, default)

    # Сохраняем конфигурацию в JSON-файл
    with open(config_path, 'w') as config_file:
        json.dump(fields, config_file, indent=4)

    print(f"Файл конфигурации успешно сохранен: {config_path}", flush=True)
    return config_path


def load_config(config_path):
    """Загружает файл конфигурации и проверяет его на корректность."""
    global fields
    try:
        with open(config_path, 'r') as config_file:
            fields = json.load(config_file)
        print(f"Файл конфигурации успешно загружен: {config_path}", flush=True)
    except FileNotFoundError:
        print(f"Файл конфигурации не найден: {config_path}", flush=True)
        return None

    # Проверка загруженной конфигурации
    check_config(fields)

    return fields

def check_config(config):
    """Проверяет конфигурацию на корректность."""
    missing_fields = []
    invalid_paths = []

    # Проверка наличия всех обязательных полей
    for key, (data_type, prompt, default) in instructions.items():
        if key not in config or config[key] is None or (isinstance(config[key], str) and not config[key].strip()):
            missing_fields.append(key)

    # Проверка путей, которые должны быть валидными
    for field in path_fields:
        if field in config and not check_file_exists(config[field]):
            invalid_paths.append(field)

    # Обработка отсутствующих и некорректных полей
    if missing_fields:
        print("Отсутствуют или не заполнены следующие поля:", flush=True)
        for field in missing_fields:
            print(f"- {field}")
        update = input("Хотите обновить/добавить эти поля? (да/нет): ").strip().lower()
        if update == "да":
            create_or_update_config()

    if invalid_paths:
        print("Следующие пути недействительны или файлы отсутствуют:", flush=True)
        for field in invalid_paths:
            print(f"- {field}")
        update = input("Хотите обновить пути? (да/нет): ").strip().lower()
        if update == "да":
            create_or_update_config()

    if not missing_fields and not invalid_paths:
        print("Конфигурация корректна.")

def main():
    print("Введите путь к файлу конфигурации или оставьте пустым для создания нового: ", flush=True)
    config_path = input("Введите путь к файлу конфигурации или оставьте пустым для создания нового: ").strip() 
    print(f"Вы ввели {config_path}", flush=True)  

    if config_path:
        # Загружаем существующую конфигурацию
        config = load_config(config_path)
        if not config:
            print("Не удалось загрузить конфигурацию. Создаем новую.", flush=True)
            config_path = create_or_update_config()
    else:
        # Создаем новую конфигурацию
        print("Создаем новую конфигурацию", flush=True)
        config_path = create_or_update_config()
        config = load_config(config_path)

    print(f"Используем конфигурацию: {config_path}", flush=True)
    return config
