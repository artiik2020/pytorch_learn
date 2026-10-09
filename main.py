import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image


# ==========================================
# ШАГ 1: Подготовка конвейера данных
# ==========================================
# 1. Говорим PyTorch превращать картинки в цифры (тензоры)
transform = transforms.Compose([transforms.ToTensor()])

#проверяем видеокарту
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Скачиваем датасет MNIST (60 тысяч фоток цифр)
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 3. Создаем курьера (DataLoader), который таскает по 32 картинки за раз
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# ==========================================
# ШАГ 2: Твоя Модель (Мозг)
# ==========================================
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 784 входа (28x28 пикселей), 128 скрытых нейронов
        self.nl1 = nn.Linear(784, 512)
        self.redu = nn.ReLU()  # Активация (фильтр отрицательных значений)
        # 128 входов, 10 выходов (цифры от 0 до 9)
        self.nl2 = nn.Linear(512, 100)

    def forward(self, x):
        x = self.nl1(x)
        x = self.redu(x)
        x = self.nl2(x)
        return x


model = Model().to(device)  # Создаем экземпляр мозга

# ==========================================
# ШАГ 3: Инспектор и Тренер
# ==========================================
loss_fn = nn.CrossEntropyLoss()  # Инспектор (считает ошибки)
optimizer = optim.AdamW(model.parameters(), lr=0.001)  # Тренер (обновляет веса)

# ==========================================
# ШАГ 4: СВЯТОЙ ЦИКЛ ОБУЧЕНИЯ
# ==========================================
model.train()  # Включаем режим "Тренировка"

for epoch in range(5):  # Учимся 3 раза прогнать весь датасет (эпохи)
    total_loss = 0

    for images, labels in train_loader:  # DataLoader выдает пачку из 32 картинок и 32 правильных ответа

        # 🔥 ГЛАВНАЯ ФИШКА: Превращаем квадратные картинки [32, 1, 28, 28] в колбасу [32, 784]
        images = images.view(images.size(0), -1)
        images = images.to(device)
        labels = labels.to(device)

        # 1. Прогоняем через мозг
        outputs = model(images)  # На выходе получаем 32 списка по 10 цифр (оценки уверенности)

        # 2. Инспектор сравнивает предсказания с реальными ответами (labels)
        loss = loss_fn(outputs, labels)

        # 3. Магия обновления весов
        optimizer.zero_grad()  # Чистим доску
        loss.backward()  # Считаем вину каждого нейрона
        optimizer.step()  # Обновляем веса

        total_loss += loss.item()

    print(f"Эпоха {epoch + 1} завершена. Средняя ошибка: {total_loss / len(train_loader):.4f}")

def load_my_image(image_path):
    img = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor

model.eval()

with torch.no_grad():
    my_image = load_my_image('my_data/img1.png')
    my_image_data = my_image.view(1, -1)
    output = model(my_image_data)
    probabilities = f.softmax(output, dim=1)

    prdictable_digit = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][prdictable_digit].item() * 100

print(f"модель думает что это {prdictable_digit}")
print(f"Уверенна на {confidence:.2f}%")
