# Flower Image Classifier

Transfer learning over 102 flower categories, exposed as two command-line tools: one to train a
checkpoint, one to predict from it.

Built for Udacity's AI Programming with Python nanodegree.

---

## Train

```bash
python train.py flowers/ --arch vgg16 --hidden_units 512 --epochs 5 --gpu
```

| Flag | Default | |
|---|---|---|
| `data_dir` | — | dataset root (positional) |
| `--arch` | `vgg16` | `vgg16` or `densenet121` |
| `--hidden_units` | `512` | width of the hidden layer |
| `--learning_rate` | `0.001` | |
| `--epochs` | `5` | |
| `--save_dir` | `.` | where the checkpoint is written |
| `--gpu` | off | use CUDA when available |

The pretrained convolutional base is frozen and a fresh classifier head is attached —
`Linear(25088 → hidden_units) → ReLU → Dropout → Linear(hidden_units → 102)`. Only the head
trains.

## Predict

```bash
python predict.py flowers/test/1/image_06743.jpg checkpoint.pth \
    --top_k 5 --category_names cat_to_name.json --gpu
```

Returns the top *K* classes with probabilities. `--category_names` maps class indices to
readable flower names via `cat_to_name.json`; without it you get indices.

## Why two entry points

Training and inference have different dependencies and different hardware needs. Keeping them as
separate CLIs means prediction runs anywhere the checkpoint can be loaded, without dragging the
training loop along — the same separation `predictor.py` exists for in the landmark classifier.

## Layout

```
train.py                     training CLI
predict.py                   inference CLI
utils.py                     image preprocessing, checkpoint save/load
cat_to_name.json             class index → flower name
Image Classifier Project.ipynb   the notebook the CLIs were factored out of
```

The dataset is not committed — point `data_dir` at a `train`/`valid`/`test` split.
