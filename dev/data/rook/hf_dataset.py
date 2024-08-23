from datasets import load_dataset
from glob import glob

ds = load_dataset("text",
        data_files={"train": glob("rook_train_709k.txt"), "test": glob("rook_val_500.txt")})

print(ds)
ds.push_to_hub("jrahn/rook")
