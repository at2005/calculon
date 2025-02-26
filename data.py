import orjson
from torch.utils.data import Dataset
import os
import torch
import tqdm
from tokenizers import ByteLevelBPETokenizer
from common import seq_len, vocab_size
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


base_dir = "/data/MathPile/train"


def process_file(file_path):
    texts = []
    with open(file_path, "rb") as file:
        for line in file:
            json_loaded = orjson.loads(line)
            if "text" in json_loaded:
                texts.append(json_loaded["text"] + "<EOS>")
    return texts

def data_iterator(base_dir):
    file_paths = []
    for source in os.listdir(base_dir):
        source_path = os.path.join(base_dir, source)
        if not os.path.isdir(source_path):
            continue
        for json_file in os.listdir(source_path):
            file_path = os.path.join(source_path, json_file)
            file_paths.append(file_path)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, fp) for fp in file_paths]
        for future in as_completed(futures):
            for text in future.result():
                yield text

    

def train_tokeniser():
    data_gen = data_iterator(base_dir)
    tokeniser = ByteLevelBPETokenizer()
    tokeniser.train_from_iterator(
        data_gen,
        vocab_size=vocab_size,
        special_tokens=["<EOS>", "<PAD>", "<BOS>", "<UNK>"]
    )

    tokeniser.save_model(".", "math_bpe")

def encode_data():
    tokeniser = ByteLevelBPETokenizer(
        "math_bpe-vocab.json",
        "math_bpe-merges.txt"
    ) 

    batch_size = 20_000
    data_gen = data_iterator(base_dir)
    all_ids = []
    batch = []
    for text in tqdm.tqdm(data_gen, disable=False, file=sys.stdout):
        batch.append(text)
        if len(batch) == batch_size:
            encoded_batch = tokeniser.encode_batch(batch)
            for enc in encoded_batch:
                all_ids.append(enc.ids)
            batch = []
    # Process any remaining texts
    if batch:
        encoded_batch = tokeniser.encode_batch(batch)
        for enc in encoded_batch:
            all_ids.append(enc.ids)
    
    flat_ids = [token for sublist in all_ids for token in sublist]
    print("Finished processing. Writing to Tensor")
    data_tensor = torch.tensor(flat_ids, dtype=torch.uint16)

    torch.save(data_tensor, os.path.join(base_dir, "data.pt"))
    print(f"Saved tokenized data")


class MathDataset(Dataset): 
    def __len__(self):
        return len(self.data_tokens) - seq_len

    def __getitem__(self, index):
        return self.data_tokens[index: index + seq_len], self.data_tokens[index + 1 : index + seq_len + 1]

    def __init__(self):
        try:
            self.data_tokens = torch.load(os.path.join(base_dir, "data.pt")).to(torch.long)
            self.tokeniser = ByteLevelBPETokenizer(
                "math_bpe-vocab.json",
                "math_bpe-merges.txt"
            )

        except FileNotFoundError:
            print("Train tokeniser first")


if __name__ == "__main__":
    train_tokeniser()
    print("Finished training tokeniser. Starting to encode...")
    encode_data()