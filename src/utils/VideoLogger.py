import numpy as np
import cv2
import random
import pandas as pd


def draw_gaze_points(frame, gaze_df, name, frame_number, supp):
    """
    Наносит точки взгляда на кадр с учетом параметров и конфигурации.

    Parameters:
        frame (np.ndarray): Текущий кадр видео.
        frame_number (int): Номер текущего кадра.
        gaze_df (pd.DataFrame): Датафрейм с данными о взглядах, ожидаются колонки 'world_index', 'norm_pos_x', 'norm_pos_y'.
        supp (dict): Словарь с конфигурацией точек, включая радиус, цвет и прозрачность.
        name (str): Имя текущей метки для конфигурации.

    Returns:
        modified_frame (np.ndarray): Кадр с нанесёнными точками.
    """

    # Проверка типа frame
    if not isinstance(frame, np.ndarray):
        raise Exception(
            "draw_gaze_points in logging function",
            f"Ошибка: frame должен быть np.ndarray, получено {type(frame)}",
        )

    # Создаем копию кадра для прозрачного наложения точек
    overlay = frame.copy()

    # Отбор данных взгляда для текущего кадра
    gaze_points = gaze_df[gaze_df["world_index"] == frame_number]

    # Проход по точкам взгляда
    for _, gaze_point in gaze_points.iterrows():
        if not np.isnan(gaze_point["norm_pos_x"]) and not np.isnan(
            gaze_point["norm_pos_y"]
        ):
            # Определение координат точки на кадре
            center_x = int(gaze_point["norm_pos_x"] * frame.shape[1])
            center_y = int((1 - gaze_point["norm_pos_y"]) * frame.shape[0])

            # Рисуем точку на наложении
            cv2.circle(
                overlay,
                (center_x, center_y),
                supp["rads"][name],
                supp["colors"][name][::-1],  # BGR вместо RGB
                -1,  # Залитая окружность
            )

    # Наложение изображения с точками на оригинальный кадр
    modified_frame = cv2.addWeighted(
        overlay, supp["alphas"][name], frame, 1 - supp["alphas"][name], 0
    )

    return modified_frame


def enlarge_mask(mask, constant):
    """
    Увеличивает маску на заданное количество пикселей с помощью морфологических операций.

    Parameters:
        mask (np.ndarray): Бинарная маска (0 или 255).
        constant (int): Величина расширения в пикселях.

    Returns:
        enlarged_mask (np.ndarray): Увеличенная маска.
    """
    # Поиск контуров
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("Контуры не найдены в маске.")
        return mask

    # Создание структурного элемента
    kernel_size = max(int(constant), 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Расширение маски
    enlarged_mask = cv2.dilate(mask, kernel, iterations=1)

    if enlarged_mask.max() == 0:
        print("Увеличенная маска пустая.")

    return enlarged_mask


def draw_segmented_mask(frame, mask, color: list, alpha):
    """
    Объединяет изображение и его сегментационную маску.

    Parameters:
        frame (np.ndarray): Исходное изображение.
        mask (np.ndarray): Сегментационная маска.
        color (list): Цвет маски в формате [R, G, B].
        alpha (float): Прозрачность маски (от 0 до 1).

    Returns:
        image_combined (np.ndarray): Изображение с наложенной маской.
    """
    # Создаем копию изображения
    image = np.copy(frame)

    # Инвертируем цвет из RGB в BGR для OpenCV
    color = color[::-1]

    # Расширение маски до 3 каналов
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)

    # Создание маскированного массива с цветом
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)

    # Получение изображения с наложенной маской
    image_overlay = masked.filled()

    # Объединение исходного изображения и наложенного
    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined


def draw_box(img, x, color=None, label=None, line_thickness=3):
    """
    Отрисовывает один ограничивающий бокс на изображении и добавляет метку.

    Parameters:
        x (list or tuple): Координаты бокса [xmin, ymin, xmax, ymax].
        img (np.ndarray): Изображение.
        color (list or tuple, optional): Цвет бокса [R, G, B], по умолчанию случайный.
        label (str, optional): Текст метки, по умолчанию отсутствует.
        line_thickness (int, optional): Толщина линии, по умолчанию 3.

    Returns:
        None: Изменяет изображение на месте.
    """
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1

    # Генерация случайного цвета, если не указан
    color = color or [random.randint(0, 255) for _ in range(3)]
    color = color[::-1]  # Инвертируем цвет для BGR
    # Координаты углов бокса
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

    # Рисуем прямоугольник
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def draw_all_segmented_masks(frame, masks, colors, alphas, rads=None):
    """
    Отрисовывает все сегментированные маски на изображении.

    Parameters:
        frame (np.ndarray): Исходное изображение.
        masks (dict): Словарь масок.
        colors (dict): Словарь цветов масок.
        alphas (dict): Словарь прозрачностей масок.
        rads (dict, optional): Радиусы для увеличения масок, по умолчанию None.

    Returns:
        image (np.ndarray): Изображение с наложенными масками.
    """
    image = np.copy(frame)

    if masks is not None:
        for mask_name in masks.keys():
            image = draw_segmented_mask(
                image, masks[mask_name], colors[mask_name], alphas
            )
            if rads:
                mask = enlarge_mask(masks[mask_name], int(rads[mask_name]))
                image = draw_segmented_mask(image, mask, colors[mask_name], alphas)

    return image


def draw_all_boxes(frame, boxes, colors, class_names):
    """
    Отрисовывает все предсказанные боксы на изображении.

    Parameters:
        frame (np.ndarray): Исходное изображение.
        boxes (list): Список предсказанных объектов.
        colors (dict): Словарь цветов боксов.
        class_names (list): Список с именами классов.

    Returns:
        image (np.ndarray): Изображение с отрисованными боксами.
    """
    image = np.copy(frame)

    for box in boxes:
        xmin = int(box.data[0][0])
        ymin = int(box.data[0][1])
        xmax = int(box.data[0][2])
        ymax = int(box.data[0][3])
        draw_box(
            image,
            (xmin, ymin, xmax, ymax),
            colors[class_names[int(box.cls)]],
            f"{class_names[int(box.cls)]} {float(box.conf):.3}",
        )

    return image


COLOR_LIB = [
    (0, 113, 188),
    (216, 82, 24),
    (236, 176, 31),
    (125, 46, 141),
    (118, 171, 47),
    (88, 24, 69),
    (45, 117, 15),
    (25, 170, 228),
    (251, 6, 250),
    (6, 251, 134),
    (253, 0, 0),
    (0, 143, 57),
]

logging_types = {
    "No logging": 0b000000,  # Без логирования
    "M": 0b000001,  # Логирование масок
    "B": 0b000010,  # Логирование боксов
    "G": 0b000100,  # Логирование точек взгляда
    "F": 0b001000,  # Логирование номера кадра
    "E": 0b010000,  # Логирование только ошибок в журнал
    "A": 0b100000,  # Логирование всех действий в журнал
    "Full": 0b111111,  # Полное логирование
}
arg_names = {
    "frame": 0b00000001,
    "masks": 0b00000010,
    "boxes": 0b00000100,
    "gaze_data": 0b00001000,
    "label": 0b00010000,
    "frame_number": 0b00100000,
    "text_log_data": 0b01000000,
    "segment_rads": 0b10000000
}


def parse_modes(**kwargs):
    """
    Парсинг типов логирования в бинарный код. Только одна обработка за раз

    Parameters:
        kwargs: Аргументы с ключом "logging_type", содержащие строку с типами логирования.

    Returns:
        cipher (int): Бинарный код, представляющий выбранные типы логирования.
    """
    if len(kwargs) != 1:
        raise Exception(
            "parse_modes in VideoLogger",
            "Попытка сломать шифратор логгера (Только один ключ за раз!)",
        )
    cipher = 0b00000000
    if "logging_type" in kwargs.keys():
        if "No logging" not in kwargs["logging_type"]:
            if "Full" not in kwargs["logging_type"]:
                for letter in kwargs["logging_type"]:
                    if letter in logging_types.keys():
                        cipher = cipher | logging_types[letter]
            else:
                cipher = logging_types["Full"]
        else:
            cipher = logging_types["No logging"]
    if "arg_names" in kwargs.keys():
        for name in kwargs["arg_names"]:
            cipher = cipher | arg_names[name]
    return cipher


class VideoLogger:
    def __init__(self, logging_type: str = "No logging", **kwargs):
        """
        Инициализация видео-логгера, который управляет визуализацией масок, боксов и точек взгляда на видео.

        Parameters:
            logging_type (str): Строка с типами логирования. Можно вписать "Full" или "No logging", а можно и
            использовать комбинации следующих символов:
                - "M": Визуализация масок.
                - "B": Визуализация боксов.
                - "G": Визуализация точек взгляда.
                - "F": Отображение номера кадра.
                - "E": Логирование только ошибок в журнал.
                - "A": Логирование всех действий в журнал.
                Пример: "MBG" для масок, боксов и точек взгляда.

            kwargs: Дополнительные параметры настройки:
                - seg_vis (dict): Параметры для визуализации масок, включая цвета и прозрачность.
                - gaze_vis (dict): Параметры для визуализации точек взгляда, включая радиусы, цвета и прозрачность.
        """

        self.logging_type = parse_modes(logging_type=logging_type)

        if not self.logging_type:
            return

        if self.logging_type & 0b110000:
            if "text_log" in kwargs.keys():
                pass
            else:
                raise Exception(
                    "init in VideoLogger",
                    "При включенном текстовом логгировании обязательно нужно указать путь к выходному файлу (text_log = path)",
                )

        # Настройка для визуализации масок
        if self.logging_type & 0b000011:
            if "seg_vis" in kwargs.keys():  # Логирование масок
                self.seg = {}
                # Настройка цветов для классов масок
                if "classes" in kwargs["seg_vis"].keys():
                    self.seg["colors"] = {}
                    for i, val in enumerate(kwargs["seg_vis"]["classes"]):
                        self.seg["colors"].update({val: COLOR_LIB[i]})
                else:
                    # Значения цветов по умолчанию
                    self.seg["colors"] = {
                        "base": (88, 24, 69),
                        "object": (45, 117, 15),
                        "hand": (25, 170, 228),
                        "target": (251, 6, 250),
                        "prosthesis": (6, 251, 134),
                    }

                # Прозрачность для масок
                self.seg["alpha"] = kwargs["seg_vis"].get("alpha", 0.3)
            else:
                raise Exception(
                    "init in VideoLogger",
                    """При включенной визуализации масок или боксов нужно указать хотябы лист названий классов 
                                (seg_vis = {"classes": list(classes.keys())})""",
                )

        # Настройка для визуализации точек взгляда
        if self.logging_type & 0b000100:
            if "gaze_vis" in kwargs.keys():  # Логирование точек взгляда
                self.gaze = {}
                # Настройка цветов для классов точек взгляда
                if "classes" in kwargs["gaze_vis"].keys():
                    self.gaze["colors"] = {}
                    for i, val in enumerate(kwargs["gaze_vis"]["classes"]):
                        self.gaze["colors"].update({val: COLOR_LIB[::-1][i]})

                    # Радиусы точек взгляда
                    self.gaze["rads"] = {}
                    if "rads" in kwargs["gaze_vis"].keys():
                        for k, v in zip(
                            kwargs["gaze_vis"]["classes"], kwargs["gaze_vis"]["rads"]
                        ):
                            self.gaze["rads"].update({k: v})
                    else:
                        for k in kwargs["gaze_vis"]["classes"]:
                            self.gaze["rads"].update({k: 15})

                    # Прозрачность точек взгляда
                    self.gaze["alphas"] = {}
                    if "alphas" in kwargs["gaze_vis"].keys():
                        for k, v in zip(
                            kwargs["gaze_vis"]["classes"], kwargs["gaze_vis"]["alphas"]
                        ):
                            self.gaze["alphas"].update({k: v})
                    else:
                        for k in kwargs["gaze_vis"]["classes"]:
                            self.gaze["alphas"].update({k: 0.5})
                else:
                    raise ValueError("Dataframes names are not stated")
            else:
                raise Exception(
                    "init in VideoLogger",
                    """При включенной визуализации взглядов нужно указать, как минимум лист названий датафреймов 
                                (gaze_viz = {"classes": list(gaze_data.keys())})""",
                )

    def log(self, **kwargs):
        """
        Логирование кадра с возможностью отрисовки масок, боксов и точек взгляда.

        Parameters:
            frame (np.ndarray): Исходный кадр.
            kwargs: Дополнительные параметры для отрисовки, такие как маски, боксы и данные о взгляде.
        """
        if not self.logging_type:
            return

        cipher = parse_modes(arg_names=list(kwargs.keys()))

        if cipher & 0b0000001:
            img = np.copy(kwargs["frame"])
            # Отрисовка сегментированных масок
            if (cipher & 0b0000010) and (
                self.logging_type & 0b000001
            ):  # Логирование масок
                if cipher & 0b10000000:
                    segment_rads = kwargs["segment_rads"]
                else:
                    segment_rads = None

                img = draw_all_segmented_masks(
                    img, kwargs["masks"], self.seg["colors"], self.seg["alpha"], segment_rads
                )

            # Отрисовка боксов
            if (cipher & 0b0000100) and (
                self.logging_type & 0b000010
            ):  # Логирование боксов

                img = draw_all_boxes(
                    img,
                    kwargs["boxes"],
                    self.seg["colors"],
                    list(self.seg["colors"].keys()),
                )

            # Отрисовка точек взгляда
            if (cipher & 0b0111000) and (
                self.logging_type & 0b000100
            ):  # Логирование точек взгляда

                img = draw_gaze_points(
                    img,
                    kwargs["gaze_data"],
                    kwargs["label"],
                    kwargs["frame_number"],
                    self.gaze,
                )

            # Отрисовка номера кадра
            if (cipher & 0b0100000) and (
                self.logging_type & 0b001000
            ):  # Логирование номера кадра
                cv2.putText(
                    img,
                    str(kwargs["frame_number"]),
                    (150, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (36, 255, 12),
                    2,
                )

            # Копирование изменений обратно в исходный кадр
            np.copyto(kwargs["frame"], img)

        # Запись в журнал
        if (cipher & 0b1000000) and (
            self.logging_type & 0b110000
        ):  # Логирование текстовой информации
            pass
