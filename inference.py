from train import Transformer, dim, num_layers
import torch
import torch.nn as nn
from tokenizers import ByteLevelBPETokenizer

inference_device = "mps"


def load_inference_transformer(checkpoint_file):
    with torch.no_grad():
        test_time_transformer = Transformer(dim, num_layers).to(inference_device)
        checkpoint = torch.load(
            checkpoint_file,
            map_location=torch.device(inference_device),
            weights_only=True,
        )

        state_dict = checkpoint["model_state_dict"]
        new_state_dict = {}

        for k, v in state_dict.items():
            if k.startswith("module._orig_mod."):
                name = k[len("module._orig_mod.") :]
            elif k.startswith("module."):
                name = k[len("module.") :]
            else:
                name = k

            new_state_dict[name] = v

        test_time_transformer.load_state_dict(new_state_dict)
        test_time_transformer.eval()
        return test_time_transformer


def get_tokenizer():
    return ByteLevelBPETokenizer("math_bpe-vocab.json", "math_bpe-merges.txt")


def tokenize_input(tokenizer, prompt):
    return torch.Tensor(tokenizer.encode(prompt).ids).to(torch.long)


def inference(test_time_transformer: nn.Module, n_tokens, prompt, print_output=False):
    tokenizer = get_tokenizer()
    input_tokens = tokenize_input(tokenizer, prompt).to(inference_device).unsqueeze(0)

    if print_output:
        print(prompt, end="")

    for _ in range(n_tokens):
        _, input_tokens = test_time_transformer(input_tokens, temperature=1.0)
        if print_output:
            decoded = tokenizer.decode(input_tokens[0:,].cpu().tolist()[0])
            print(decoded, end="")


if __name__ == "__main__":
    transformer = load_inference_transformer("checkpoint_2_870.pt")
    inference(transformer, 500, "Heisenberg", print_output=True)
