import cv2
from ..utils.VideoLogger import VideoLogger


class VideoAggregation:
    def __init__(
        self,
        input_path: str,
        output_path: str,
        Logger,
        frame_floor: int = 0,
        frame_cap: int = None,
    ):
        """
        Инициализирует класс для обработки видео, позволяя чтение и запись кадров.

        Params:
            input_path (str): Путь к входному видеофайлу.
            output_path (str): Путь к выходному видеофайлу, где будут записаны обработанные кадры.
            frame_cap (int, optional): Максимальное количество кадров для обработки. По умолчанию None.
        """
        self.end = False  # Флаг, указывающий на завершение обработки видео
        self.frame_counter = -1 # Счетчик кадров
        self.frame_cap = frame_cap  # Максимальное количество кадров для обработки
        self.frame_floor = frame_floor  # Кадр начала записи
        self.input_video = cv2.VideoCapture(input_path)  # Открываем входное видео
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Кодек для сохранения выходного видео
        self.output_video = cv2.VideoWriter(
            output_path, fourcc, 30.0, (1920, 1080)
        )  # Создаем выходной видеофайл
        self.Logger = Logger

    def read(self):
        """
        Читает следующий кадр из входного видео.

        Returns:
            frame (np.ndarray): Кадр, считанный из видео. Если видео закончилось, возвращает None.
        """
        ret, frame = self.input_video.read()  # Читаем кадр
        self.end = not ret  # Обновляем флаг завершения
        self.frame_counter += 1  # Увеличиваем счетчик кадров
        return frame  # Возвращаем считанный кадр

    def write(self, frame):
        """
        Записывает кадр в выходное видео.

        Params:
            frame (np.ndarray): Кадр, который будет записан в выходное видео.
        """
        if not int(self.Logger.logging_type):
            print(int(self.Logger.logging_type))
            return
        if self.already():
            self.output_video.write(frame)  # Записываем кадр в выходное видео

    def enough(self):
        """
        Проверяет, достигнуто ли максимальное количество обрабатываемых кадров.

        Returns:
            bool: True, если количество обработанных кадров превышает frame_cap, иначе False.
        """
        return (
            False if self.frame_cap is None else self.frame_cap <= self.frame_counter
        )  # Сравниваем с лимитом кадров

    def already(self):
        """
        Проверяет, достигнуто ли минимальное количество обрабатываемых кадров.

        Returns:
            bool: True, если количество обработанных кадров превышает frame_floor, иначе False.
        """
        return self.frame_floor <= self.frame_counter  # Сравниваем с порогом кадров

    def release(self):
        """
        Освобождает ресурсы, используемые для обработки видео.
        """
        self.input_video.release()  # Закрываем входное видео
        self.output_video.release()  # Закрываем выходное видео
