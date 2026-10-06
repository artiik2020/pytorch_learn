# Создаем тензор (он по умолчанию живет в CPU, в оперативной памяти)
import torch

x = torch.tensor(4.0, requires_grad=True)

# Проверяем, есть ли у нас видеокарта от NVIDIA
if torch.cuda.is_available():
    y = x * 3
    y.backward()
    print(x.grad)
else:
    print("Видеокарты нет, придется страдать на CPU 😢")
