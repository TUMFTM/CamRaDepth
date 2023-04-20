from tqdm import tqdm

def create_tqdm_bar(iterable, desc="Progress"):
    return tqdm(enumerate(iterable),total=len(iterable), ncols=150, desc=desc)
