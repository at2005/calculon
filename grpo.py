from models import Transformer
from common import dim, num_layers, vocab_size
import torch
from inference import tokenize_input, get_tokenizer
from tqdm import tqdm


# inference time generation
MAX_SEQ_LEN = 1000
NUM_ITERS = 100
NUM_STEPS = 200
NUM_GRPO_ITERS = 10
num_groups = 128
group_size = 64
num_tokens = 300
epsilon = 0.2
beta = 0.3
lr = 3e-4


# stores a tensor of prior values
def generate_group(transformer: Transformer, encoded_prompt: torch.Tensor):
    # encoded prompt is B, G, T_init
    token_to_pass = encoded_prompt
    tokens = encoded_prompt
    priors = []
    # token generation step, we now just output a bunch of tokens
    for t in range(num_tokens):
        prior, token_to_pass = transformer(token_to_pass, output_programs=True)
        tokens = torch.cat([tokens, token_to_pass], dim=-2)
        priors.append(prior)
    priors = torch.stack(priors)
    # tokens are now B,G,T, priors are B, G, T, V
    return priors, tokens


def grader(group_outputs: torch.Tensor):
    # grade generations
    # return some list of advantages
    # group output is B, G, T
    # grades are B, G
    B, G, _ = group_outputs.shape
    return torch.zeros(B, G)


def get_advantages(tokens) -> torch.Tensor:
    with torch.no_grad():
        rewards: torch.Tensor = grader(
            tokens
        )  # rewards of dimension (B, G), ie one for each term in the group for each batch
        mean_reward = torch.mean(rewards, dim=-1)  # across each group
        reward_std = torch.std(rewards, dim=-1)  # across each group
        advantages = (rewards - mean_reward) / (reward_std + 1e-8)  # get (B, G)
        advantages = advantages.unsqueeze(-1)  # (B, G, 1)
        return advantages


def grpo_loss(advantages, new_priors, old_priors, reference_priors, token_ids):
    # new priors are of shape (B, G, T, V)
    ratio = new_priors[:, :, token_ids, :] / old_priors[:, :, token_ids, :]
    min_clipped_advantage = torch.min(
        advantages * ratio,  # B, G, 1 *  B, G, T checks out
        torch.clip(ratio, 1 - epsilon, 1 + epsilon) * advantages,
    )

    ref_ratio = reference_priors[:, :, token_ids, :] / new_priors[:, :, token_ids, :]
    kl_loss = ref_ratio - torch.log(ref_ratio) - 1.0  # B, G, T, 1
    objective: torch.Tensor = min_clipped_advantage - beta * kl_loss

    # mean across T, across G, and B
    loss = -objective.mean(dim=[0, 1, 2])
    return loss


# must also implement padding
def encode_prompt(tokenizer, prompt) -> torch.Tensor:
    token_tensor = torch.empty(num_groups, group_size, MAX_SEQ_LEN, dim)
    for group in range(len(prompt)):
        for item in range(len(prompt[group])):
            tokenised = tokenize_input(tokenizer, item)
            token_tensor[group, item, :, :] = tokenised
    return token_tensor


def train_grpo_custom():
    policy = Transformer(dim=dim, num_layers=num_layers)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, betas=(0.9, 0.95))
    tokenizer = get_tokenizer()
    for _i in tqdm(range(NUM_ITERS)):
        reference = Transformer(dim=dim, num_layers=num_layers)
        reference.load_state_dict(policy.state_dict())
        for _j in tqdm(range(NUM_STEPS)):

            def prompt_sampler():
                return ""

            prompt_sample = prompt_sampler()  # B, G, T_variable
            encoded_prompt = encode_prompt(tokenizer, prompt_sample)  # B, G, T_fixed
            new_priors, tokens = generate_group(policy, encoded_prompt)
            advantages = get_advantages(tokens)

            with torch.no_grad():
                ref_priors, _ = generate_group(reference, encoded_prompt)

            old_priors = new_priors.detach()

            for _grpo_iter in tqdm(range(NUM_GRPO_ITERS)):
                new_priors, _ = generate_group(policy, encoded_prompt)
                optimizer.zero_grad()
                loss = grpo_loss(advantages, new_priors, old_priors, ref_priors)
                loss.backward()
                optimizer.step()
