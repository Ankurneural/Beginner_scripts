
import pickle
import gzip

from pathlib import Path
import requests

data_path = Path("data")
path = data_path/"mnist"
path.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (path/FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (path/FILENAME).open("wb").write(content)


with gzip.open((path/FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

from matplotlib import pyplot

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
# print(x_train.shape)