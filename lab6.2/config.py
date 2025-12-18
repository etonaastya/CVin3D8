import torch

class Config:
    DATA_ROOT = "C:/Users/anast/ml/CVin3D/lab4/task2/Stanford3dDataset"          
    CHECKPOINT_DIR = "./checkpoints"
    LOG_DIR = "./logs"

    NUM_CLASSES = 13
    CLASS_NAMES = [
        "ceiling", "floor", "wall", "beam", "column", "window", "door",
        "table", "chair", "sofa", "bookcase", "board", "clutter"
    ]
    NUM_POINTS = 4096          
    #NUM_POINTS = 9000            
    TRAIN_AREAS = ["Area_1", "Area_2", "Area_3", "Area_4", "Area_6"]
    VAL_AREA = "Area_5"
    TEST_AREA = "Area_5"       

    # Аугментация
    AUGMENT = True
    ROTATE_Z = True
    JITTER = True
    JITTER_SIGMA = 0.01
    JITTER_CLIP = 0.05
    DROPOUT_POINTS = 0.95      

    BATCH_SIZE = 8
    EPOCHS = 50
    LR = 0.001
    LR_STEP = 20
    LR_GAMMA = 0.5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    RADIUS_LIST = [0.1, 0.2, 0.4, 0.8]
    SAMPLE_RATIO_LIST = [0.5, 0.25, 0.125, 0.0625]
    M_GROUP = [32, 32, 32, 32]  # количество соседей в grouping