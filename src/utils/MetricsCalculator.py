import os
import cv2
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def get_mask_for_frame(mask_destination):
    """
    Функция для получения истинной маски для определенного фрейма

    Param:
    frame_counter: номер фрейма
    masks_folder: путь к истинным маскам

    Return:
    mask - истинная маска определенного фрейма
    """
    # Проверка, существует ли файл маски
    if not os.path.exists(mask_destination):
        raise FileNotFoundError(f"Файл маски {mask_destination} не найден")

    # Читаем маску в градациях серого
    mask = cv2.imread(mask_destination, cv2.IMREAD_GRAYSCALE)

    return mask

def get_image_filenames(folder_path):
    """
    Функция для получения списка имен файлов из папки

    Param:
    folder_path: путь к папке с изображениями

    Return:
    image_files: отсортированный список имен файлов без пути
    """
    image_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
    return sorted(image_files)


def find_common_filenames(folder1, folder2):
    """
    Функция для нахождения общих имен файлов между двумя папками

    Param:
    folder1: путь к первой папке с изображениями
    folder2: путь ко второй папке с изображениями

    Return:
    common_filenames: отсортированный список общих файлов
    """
    files1 = get_image_filenames(folder1)
    files2 = get_image_filenames(folder2)

    # Найдем пересечение названий файлов
    common_filenames = sorted(set(files1).intersection(files2))

    return common_filenames


def load_images_by_filenames(folder, filenames):
    """
    Функция для загрузки изображений по списку имен файлов

    Param:
    folder: путь к папке с изображениями
    filenames: список имен файлов

    Return:
    images: массив загруженных изображений
    """
    images = []
    for filename in filenames:
        img_path = os.path.join(folder, filename)
        img = get_mask_for_frame(img_path)
        if img is not None:
            images.append(img)
        else:
            print(f"Не удалось загрузить изображение: {filename}")

    return images

def calculate_metrics(y_trues, y_preds,  classes, background_class=0):
    # Инициализируем списки для хранения метрик
    precisions = []
    recalls = []
    f1_scores = []
    ious = []
        # Проходим по каждой паре масок
    for y_true, y_pred in zip(y_trues, y_preds):
        # Проверяем, что размеры масок совпадают
        if y_true.shape != y_pred.shape:
            print(f"Размеры масок не совпадают! Истинная: {y_true.shape}, Предсказанная: {y_pred.shape}")
            continue  # Пропускаем эту пару
    # Проходим по каждой паре масок
    for y_true, y_pred in zip(y_trues, y_preds):
        # Приводим маски к формату 1D
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()


        # Рассчитываем precision, recall, F1-score с учетом всех классов, кроме фона
        precision = precision_score(y_true_flat, y_pred_flat, average='weighted', labels=np.unique(y_true_flat[y_true_flat != background_class]), zero_division=0)
        recall = recall_score(y_true_flat, y_pred_flat, average='weighted', labels=np.unique(y_true_flat[y_true_flat != background_class]), zero_division=0)
        f1 = f1_score(y_true_flat, y_pred_flat, average='weighted', labels=np.unique(y_true_flat[y_true_flat != background_class]), zero_division=0)
        iou = jaccard_score(y_true_flat, y_pred_flat, average='macro')
        
        # Рассчитываем IoU (Intersection over Union) с учетом фона
        # intersection = np.logical_and(y_true_flat, y_pred_flat).sum()
        # union = np.logical_or(y_true_flat, y_pred_flat).sum()
        # iou = intersection / union if union != 0 else 0

        # Сохраняем результаты
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        ious.append(iou)

    # Выводим средние метрики
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    avg_f1 = np.mean(f1_scores) if f1_scores else 0
    avg_iou = np.mean(ious) if ious else 0

    print("Метрики по Weighted")
    print('\n')
    print("Precision:", avg_precision)
    print("Recall:", avg_recall)
    print("F1-Score:", avg_f1)
    print('\n')
    print("IoU:", avg_iou)
    print('\n')

    # Для визуализации матрицы путаницы
    # Соединяем все истинные и предсказанные маски
    y_true_all = np.concatenate([y.flatten() for y in y_trues])
    y_pred_all = np.concatenate([y.flatten() for y in y_preds])

    # Выводим отчет по классам, исключая фон
    print("Classification Report (excluding background):")
    print(classification_report(y_true_all, y_pred_all, target_names=classes,labels=np.unique(y_true_all[y_true_all != background_class])))

    # Рассчитываем матрицу путаницы
    conf_matrix = confusion_matrix(y_true_all, y_pred_all, normalize='true', labels=np.unique(y_pred_all[y_pred_all != background_class]))

    # Визуализируем матрицу путаницы
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', 
                xticklabels=classes, #np.unique(y_pred_all[y_pred_all != background_class]), 
                yticklabels=classes) #np.unique(y_true_all[y_true_all != background_class]))
    plt.xlabel('Предсказанные метки')
    plt.ylabel('Истинные метки')
    plt.title('Матрица путаницы (исключая фон)')
    plt.show()

    return avg_precision, avg_recall, avg_f1, avg_iou