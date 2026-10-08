from torch import nn
import torch


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.nl1 = nn.Linear(7840, 1280)
        self.redu = nn.ReLU()
        self.nl2 = nn.Linear(1280, 10)

    def forward(self, x):
        x = self.nl1(x)
        x = self.redu(x)
        x = self.nl2(x)
        return x


# Создаем модель из примера
model = Model()

# Считаем общее количество параметров
total_params = sum(p.numel() for p in model.parameters())
print(f"Всего параметров: {total_params}")

# А теперь считаем ТОЛЬКО обучаемые параметры (на случай если часть заморожена)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Обучаемых параметров: {trainable_params}")