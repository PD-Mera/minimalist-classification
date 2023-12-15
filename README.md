# Minimalist Classification Pipeline

``` bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

# Run

To train, simply run

``` bash
python runner.py
```

To visualize tensorboard, run

``` bash
tensorboard --host 0.0.0.0 --port 12345 --logdir runs
```