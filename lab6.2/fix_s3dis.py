import os
import re

# Список проблемных файлов
bad_files = [
    r"C:/Users/anast/ml/CVin3D/lab4/task2/Stanford3dDataset/Area_1/WC_1/WC_1.txt",
    r"C:/Users/anast/ml/CVin3D/lab4/task2/Stanford3dDataset/Area_6/hallway_2/hallway_2.txt",
]

def fix_line(line):
    # Заменяем любые control chars на пробел
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', line)
    # Сжимаем несколько пробелов в один
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    # Убираем точки в целых числах RGB (иногда бывает 100.0 → должно быть 100)
    parts = cleaned.split()
    if len(parts) >= 7:
        # x y z r g b label
        try:
            # Приводим к float, потом int для RGB и label
            xyz = [float(x) for x in parts[:3]]
            rgb = [int(round(float(x))) for x in parts[3:6]]
            label = int(round(float(parts[6])))
            # Ограничиваем RGB [0,255]
            rgb = [max(0, min(255, c)) for c in rgb]
            return f"{xyz[0]} {xyz[1]} {xyz[2]} {rgb[0]} {rgb[1]} {rgb[2]} {label}\n"
        except Exception as e:
            print(f"Can't fix line: '{line.strip()}' → {e}")
            return None
    return None

for path in bad_files:
    if not os.path.exists(path):
        print(f"Skip {path} — not found")
        continue
    print(f"Fixing {path}...")
    fixed_lines = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f, 1):
            fixed = fix_line(line)
            if fixed is not None:
                fixed_lines.append(fixed)
            else:
                print(f" Line {i} skipped: {line.strip()[:50]}...")
    
    backup = path + ".backup"
    if not os.path.exists(backup):
        os.rename(path, backup)
    with open(path, 'w') as f:
        f.writelines(fixed_lines)
    print(f"Fixed {len(fixed_lines)} lines → saved to {path}")