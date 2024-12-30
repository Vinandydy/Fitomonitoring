import torch
from pathlib import Path


# Метод для сохранения модели
def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    # Создаем целевую папку для сохранения
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Создаем путь к ней
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Сохраняем state_dict() модели (её веса на нейронах)
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)