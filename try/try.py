import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
from data import get_split_dataset
train_set, val_set, test_set = get_split_dataset("fluid_shake", datadir='/home/htxue/data_Pour/', want_split="all", training=True)
# train_set, val_set, test_set = get_split_dataset("srn", datadir='/home/htxue/srn_car/cars', want_split="all", training=True)

print(train_set.__getitem__(0))
