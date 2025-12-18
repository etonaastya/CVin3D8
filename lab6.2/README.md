##PointNet++: 3D Object Segmentation 

### Создание виртуального окружения
1. Создание виртуального окружения
```bash
# Создание виртуального окружения
python -m venv labs

# Активация (выберите соответствующую команду для вашей ОС)
labs\Scripts\activate  # Windows

source labs/bin/activate  # Linux/MacOS
```

3. Установка зависимостей
```bash
pip install -r requirements.txt
```

4. Подготовка данных
Указать в файле config.py путь к датасету S3DIS в DATA_ROOT

5. Обучение модели
```bash
python train.py
```

6. Оценка и визуализация
```bash
python evaluate.py
```

Что делает evaluete.py? 
Загружает best_model.pth
Считает:
Overall Accuracy, mean IoU (mIoU), IoU по каждому классу

Если вдруг будет ошибка Error loading ... \x10 ... (битые файлы в датасете), то необходимо выполнить следующую команду:
```bash
python fix_s3dis.py
```