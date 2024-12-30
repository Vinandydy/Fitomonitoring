import os
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import zipfile
from pathlib import Path
import requests
from typing import List


#Получение содержимого в каталоге
def get_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


# Создание графика обучения модели
def Graph_Create(model: torch.nn.Module,
                           X: torch.Tensor,
                           y: torch.Tensor):
    # Переводим всё на ЦП (работает лучше с numpy и matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Создаем график
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101),
                         np.linspace(y_min, y_max, 101))

    #Распологаем информацию, представленную на графике
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Делаем инференс на модели
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    #Проверка на мультиклассовость (при необходимости расширения, для того чтобы не сломать график, происходит проверка, больше ли 2-ух классов у модели)
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Получаем результаты и размещаем на графике
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


# Метод для размещения линейной информации на графике
def Graph_Info(
        train_data, train_labels, test_data, test_labels, predictions=None
):
    # Создаём график
    plt.figure(figsize=(10, 7))
    #Тренеровка
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    #Тесты
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Размещаем предсказания
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Размещаем ключи для понимания графика на нём же
    plt.legend(prop={"size": 14})


# Рассчитываем точность предсказаний
def accuracy_fn(y_true, y_pred):
    # Суммируем количество правильных предсказаний
    correct = torch.eq(y_true, y_pred).sum().item()
    # Рассчитываем количество верных всех предсказаний в процентном соотношении
    acc = (correct / len(y_pred)) * 100
    return acc


# Расчёт времени, которое ушло на тренировку
def print_train_time(start, end, device=None):
    total_time = end - start
    print(f"\nТренировка была проедена на {device} за: {total_time:.1f} секунд")
    return total_time


# Метод для создания графика потерь
def Graph_loss(results):
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Размещаем потери
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Размещаем точность
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


# Метод для инференса и размещения результатов на графике
def Graph_result(
        model: torch.nn.Module,
        image_path: str,
        class_names: List[str] = None,
        transform=None,
        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    # Загружаем изображение и переводим его в float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    target_image = target_image / 255.0

    if transform:
        target_image = transform(target_image)
    model.to(device)

    model.eval()
    with torch.inference_mode():
        # Добавляем ещё одно измерение изображения
        target_image = target_image.unsqueeze(dim=0)
        # Отправляем изображение на инференс модели
        target_image_pred = model(target_image.to(device))

    #Конвертируем лоджиты в уверенность модели в предсказаниях
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    #Конвертируем уверенность в лейбл класса
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    #Размещаем результаты на графике
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)


# Метод для ручного ввода seed'ов
def set_seeds(seed: int = 42):
    # Сид для всех операций pytorch
    torch.manual_seed(seed)
    # Сид для CUDA ядер
    torch.cuda.manual_seed(seed)


# Метод для загрузки данных
def download_data(source: str,
                  destination: str,
                  remove_source: bool = True) -> Path:
    # Подготавливаем папку для данных
    data_path = Path("data/")
    image_path = data_path / destination

    # Если её не существует, то создаем
    if image_path.is_dir():
        print(f"[Инфо] {image_path} Существует, создание не требуется.")
    else:
        print(f"[Инфо] Не наедена{image_path} Создаем новую.")
        image_path.mkdir(parents=True, exist_ok=True)

        # Загружаем данные из источника
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[Инфо] Загрузка {target_file} Из {source}...")
            f.write(request.content)

        # Вытаскиваем из архива
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[Инфо] Распаковка {target_file} Информации...")
            zip_ref.extractall(image_path)

        # Удаляем архив
        if remove_source:
            os.remove(data_path / target_file)

    # Возвращаем путь к данным
    return image_path
