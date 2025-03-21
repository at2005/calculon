from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from common import num_program_tokens
from grpo import (
    lr,
    NUM_GRPO_ITERS,
    NUM_ITERS,
    NUM_STEPS,
    MAX_SEQ_LEN,
    group_size,
    grpo_loss,
    get_advantages,
)
import torch
import torch.nn as nn
import tqdm

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")


def get_model(src=None):
    if src:
        config = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B")
        model = AutoModelForCausalLM.from_config(config)
        model.load_state_dict(src.state_dict())
    else:
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

    original_embeddings = model.get_input_embeddings()
    embedding_dim = original_embeddings.weight.shape[1]
    new_embeddings = torch.randn(num_program_tokens, embedding_dim) * 0.02
    extended_embeddings = torch.cat(
        [original_embeddings.weight.data, new_embeddings], dim=0
    )

    model.resize_token_embeddings(len(tokenizer) + num_program_tokens)
    model.get_input_embeddings().weight.data = extended_embeddings
    model.tie_weights()

    return model


prompt_groups = [
    "The capital of France is",
    "Shakespeare wrote",
    "Jane Austen wrote",
    "Leo Tolstoy wrote",
    "The boiling point of water is",
    "The speed of light is",
    "The closest star to Earth is",
]


def process_prompts(model, prompts):
    inputs = tokenizer(prompts, padding=True, return_tensors="pt", truncation=True)
    batch_size = len(prompt_groups)

    outputs = model.generate(
        input_ids=inputs,
        max_length=500,
        num_return_sequences=group_size,
        return_dict_in_generate=True,
        output_scores=True,
    )

    output_sequence = outputs.sequences
    output_sequence_group = output_sequence.reshape(batch_size, group_size, -1)
    scores_group = outputs.scores.reshape(batch_size, group_size, -1)
    return output_sequence_group, scores_group


def sample_batch():
    pass


def get_priors(transformer, input):
    pass


def train_grpo_llama():
    policy = get_model()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, betas=(0.9, 0.95))
    for _i in tqdm(range(NUM_ITERS)):
        reference = get_model(src=policy)
        for _j in tqdm(range(NUM_STEPS)):
            input_group = sample_batch()  # B, G, T_variable
            new_priors, tokens = get_priors(policy, input_group)
            advantages = get_advantages(tokens)

            with torch.no_grad():
                ref_priors, _ = get_priors(reference, input_group)

            old_priors = new_priors.detach()

            for _ in tqdm(range(NUM_GRPO_ITERS)):
                new_priors, tokens = get_priors(policy, input_group)
                optimizer.zero_grad()
                loss = grpo_loss(advantages, new_priors, old_priors, ref_priors, tokens)
                loss.backward()
                optimizer.step()
