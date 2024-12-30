import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

#Метод для проведения одного шага тренировки
def train_step(model: torch.nn.Module,
         dataloader: torch.utils.data.DataLoader,
         loss_fn: torch.nn.Module,
         optimizer: torch.optim.Optimizer,
         device: torch.device) -> Tuple[float, float]:

    model.train()
    #Объявляем переменные, которые будут отслеживать результаты тренеровки
    train_loss, train_acc = 0, 0

    #Засунем в пакеты данные из даталоадера и отправим на девайс
    for batch, (X, y) in enumerate(dataloader): #Для правильной расфасовки модели мы "закортежим" данные
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        #Производим расчет потерь

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        #Переводи оптимизатор в режим zero_grad (зануляем, иначе построение может быть неверным из-за старого обучения)
        optimizer.zero_grad()

        #Обратная пропагация ошибки
        loss.backward()

        #Наш шаг
        optimizer.step()

        #После прохода очередного шага, для наглядного представления, выводим полученные метрики по пакетам
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred) #Расчет вероятностей (кол-во верных обнаружений)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    #полученные метрики
    return train_loss, train_acc

#Метод для проведения одного шага теста
def test_step(model: torch.nn.Module,
         dataloader: torch.utils.data.DataLoader,
         loss_fn: torch.nn.Module,
         device: torch.device) -> Tuple[float, float]:

    model.eval()

    #Переменные, отслеживающие результат тренеровки

    test_loss, test_acc = 0, 0

    #Режим инференса модели
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)
            #Рассчет потерь
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            #Расчет точности модели (тестовый)
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    #Получаем среднее значение
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_acc

#Полная тренировка модели
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epoch: int,
          device: torch.device) -> Dict[str, List]:

    #Словарь, в котором будут вписаны результаты
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    #Отправка модели на целевой девайс
    model.to(device)

    #Цикл для прохода тренировочного и тестировочного шага
    for epoch in tqdm(range(epoch)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        #Результаты очередной эпохи
        print(
            f"Epoch: {epoch + 1} | "
            f"Train Loss: {train_loss:.3f} | "
            f"Train Acc: {train_acc:.3f} | "
            f"Test Loss: {test_loss:.3f} |"
            f"Test Acc: {test_acc:.3f}"
        )

        #Обновляем словарь со значениями (В дальнейшем понадобится для построения графа)
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)

    return results


