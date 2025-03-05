from train import Transformer, dim
import torch
from tokenizers import ByteLevelBPETokenizer


def inference(checkpoint_file, device, n_tokens, prompt):
    with torch.no_grad():
        test_time_transformer = Transformer(dim).to(device)
        checkpoint = torch.load(checkpoint_file, map_location=torch.device(device))

        state_dict = checkpoint["model_state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v

        test_time_transformer.load_state_dict(new_state_dict)
        test_time_transformer.eval()

        tokenizer = ByteLevelBPETokenizer("math_bpe-vocab.json", "math_bpe-merges.txt")
        input_tokens = (
            torch.Tensor(tokenizer.encode(prompt).ids).to(torch.long).to(device)
        ).unsqueeze(0)

        print(prompt, end="")
        for token in range(n_tokens):
            input_tokens: torch.Tensor = test_time_transformer(
                input_tokens, temperature=0.7
            )
            decoded = tokenizer.decode(input_tokens[0:,].cpu().tolist()[0])
            print(decoded, end="")


if __name__ == "__main__":
    inference("checkpoint_7_200.pt", "mps", 2000, "The theorem states that ")
