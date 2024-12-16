python mw_main/train_bc_mw.py \
    --dataset.path Assembly \
    --dataset.num_data 100 \
    --num_epoch 100 \
    --seed 1

python mw_main/train_bc_mw.py \
    --dataset.path BoxClose \
    --num_epoch 100 \
    --seed 1

python mw_main/train_bc_mw.py \
    --dataset.path StickPull \
    --dataset.num_data 100 \
    --num_epoch 100 \
    --seed 1

python mw_main/train_bc_mw.py \
    --dataset.path CoffeePush \
    --dataset.num_data 100 \
    --num_epoch 100 \
    --seed 1

python mw_main/train_bc_mw.py \
    --dataset.path ButtonPress \
    --dataset.num_data 5 \
    --num_epoch 2 \
    --seed 5