# PointNet: 3D Object Classification on ModelNet10python -m venv labs

### 1. Создание виртуального окружения

```bash
# Создание виртуального окружения
python -m venv labs

# Активация (выберите соответствующую команду для вашей ОС)
labs\Scripts\activate  # Windows

source labs/bin/activate  # Linux/MacOS
```
  
### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```
### 3. Подготовка данных

Убедитесь, что:
Распакованный ModelNet10 лежит в ../ModelNet10/
Запустите препроцессинг:
```bash
cd lab5
python data.py
```
### 4. Обучение модели
```bash
python train.py
```