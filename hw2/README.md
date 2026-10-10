# HW2 — Super-Convergence (Smith, Topin, 2018)

Воспроизведение статьи *Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates*
(arXiv:1708.07120) на CIFAR-10 / ResNet-56.

- `slides/hw2_super_convergence.pptx` — презентация
- `report/report.pdf` — отчёт
- `results/` — логи всех запусков (`<run>/log.csv`, `summary.json`), графики в `figures/`, сводка в `summary.md`

Считалось на MacBook Air M5 (PyTorch, MPS). Одна эпоха ResNet-56 ~35 с, все запуски вместе ~10 часов.

## Как запустить

```
pip install torch torchvision numpy pandas matplotlib pyarrow pillow tabulate
python prepare_data.py     # качает CIFAR-10 (с HF, с cs.toronto.edu очень медленно) -> data/cifar10.npz
./run_all.sh               # все запуски, уже посчитанные пропускает
python plots.py            # графики и таблица
```

Один запуск, например 1cycle на 50 эпох:

```
python train.py --name 1cycle_e50 --policy 1cycle --lr-min 0.1 --lr-max 3 --mom-max 0.95 --mom-min 0.85 --epochs 50
```

Политики LR: `step`, `const`, `range` (lr range test), `triangular`, `1cycle`. Остальные флаги — в `train.py`.
На колабе то же самое — `hw2_colab.ipynb`.
