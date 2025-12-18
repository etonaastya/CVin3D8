PointNet++: 3D Object Segmentation 

1. Создание виртуального окружения
# Создание виртуального окружения
python -m venv labs

# Активация (выберите соответствующую команду для вашей ОС)
labs\Scripts\activate  # Windows
source labs/bin/activate  # Linux/MacOS

2. Установка зависимостей
pip install -r requirements.txt

3. Подготовка данных
Указать в файле config.py путь к датасету S3DIS в DATA_ROOT

4. Обучение модели
python train.py

5. Оценка и визуализация
python evaluate.py

Что делает evaluete.py? 
Загружает best_model.pth
Считает:
Overall Accuracy, mean IoU (mIoU), IoU по каждому классу

Если вдруг будет ошибка Error loading ... \x10 ... (битые файлы в датасете), то необходимо выполнить следующую команду:
python fix_s3dis.py
