#!/bin/bash
# все запуски из отчёта; на M5 Air ~10 часов, на T4 ~4.
# уже посчитанные (есть summary.json) пропускаются, можно перезапускать после обрыва
set -e
cd "$(dirname "$0")"
[ -f ../.venv/bin/activate ] && source ../.venv/bin/activate
python prepare_data.py
run() { name=$1; shift; if [ -f results/$name/summary.json ]; then echo "skip $name"; else python train.py --name $name "$@"; fi; }

CYC="--policy 1cycle --lr-min 0.1 --lr-max 3 --mom-max 0.95 --mom-min 0.85"

# lr range test, lr 0->3 за 10 эпох, разные batch
run range_bs512  --policy range --lr-max 3 --epochs 10 --bs 512  --eval-every 10
run range_bs128  --policy range --lr-max 3 --epochs 10 --bs 128  --eval-every 40
run range_bs1000 --policy range --lr-max 3 --epochs 10 --bs 1000 --eval-every 5

# оценка lr по формуле (8), первые 300 итераций
run hf_const  --policy const --lr 0.1 --iters 300 --eval-every 300 --hf-estimate 300
run hf_clr    $CYC --iters 2000 --eval-every 2000 --hf-estimate 300

# основное сравнение
run 1cycle_e50  $CYC --epochs 50
run step_e50    --policy step --lr 0.35 --epochs 50
run 1cycle_e25  $CYC --epochs 25
run 1cycle_e100 $CYC --epochs 100
run step_e100   --policy step --lr 0.35 --epochs 100

# мало данных
run sub10k_1cycle --n-train 10000 $CYC --iters 1500 --eval-every 50
run sub10k_step   --n-train 10000 --policy step --lr 0.35 --iters 3000 --eval-every 50
run sub20k_1cycle --n-train 20000 $CYC --iters 1500 --eval-every 50
run sub20k_step   --n-train 20000 --policy step --lr 0.35 --iters 3000 --eval-every 50

run step_e200   --policy step --lr 0.35 --epochs 200

# weight decay
run 1cycle_e25_wd1e-3 $CYC --epochs 25 --wd 1e-3
run 1cycle_e25_wd1e-5 $CYC --epochs 25 --wd 1e-5

# максимальный lr, обычный baseline с lr 0.1, adam
run 1cycle_e25_lr1   --policy 1cycle --lr-min 0.1 --lr-max 1 --mom-max 0.95 --mom-min 0.85 --epochs 25
run 1cycle_e25_lr0.3 --policy 1cycle --lr-min 0.03 --lr-max 0.3 --mom-max 0.95 --mom-min 0.85 --epochs 25
run step_e50_lr0.1   --policy step --lr 0.1 --epochs 50
run adam_1cycle_e25 --optimizer adam --policy 1cycle --lr-min 0.0001 --lr-max 0.003 --mom-max 0.95 --mom-min 0.85 --epochs 25
echo ALL DONE
