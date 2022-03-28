# ROC Visualisation

Receiver Operator Characteristic (ROC) visualisation tool (webapp).

## Run

Run with Python:

```bash
# Optional - use virtual environment
$ python3 -m venv venv
$ source venv/bin/activate
# Install dependencies
$ pip install -r requirements.txt
# Run
$ ENV=dev python3 app.py
```

Then open in browser (URL is printed to output).

## Dev

`app.py` contains the UI build with [Dash](https://plotly.com/dash/). `roc.py` contains methods for visualisation (using [Plotly](https://plotly.com/python/)) and computation (manual and [scikit-learn](https://scikit-learn.org/stable/)).

### Deploy

```bash
$ git push heroku main
```

### @TODO

- `fixed_treshold_graph` function in `roc.py`
- introduction (model, prediction values, notation, ...)

