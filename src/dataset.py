import torch
import torch.nn.functional as F


def zero_pad_sequences(
    sequences: list[torch.Tensor],
    side: str = "left",
    value: int = 0,
    stack: bool = True,
) -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    if stack:
        return torch.stack(padded_sequences, dim=0)
    else:
        return torch.cat(padded_sequences, dim=0)


class SFTDataSet:
    def __init__(self, data, num_proc, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        system_prompt, assistant = self.data[index][:2], self.data[index][2:]

        system_prompt_templated = self.tokenizer.apply_chat_template(
            system_prompt, tokenize=True, add_generation_prompt=True
        )
        assistant_templated = self.tokenizer.apply_chat_template(
            assistant, tokenize=True
        )
        input_ids = (
            system_prompt_templated["input_ids"] + assistant_templated["input_ids"]
        )
        attention_mask = (
            system_prompt_templated["attention_mask"]
            + assistant_templated["attention_mask"]
        )
        loss_mask = [0] * len(system_prompt_templated["input_ids"]) + [1] * len(
            assistant_templated["input_ids"]
        )

        return (
            torch.tensor(input_ids),
            torch.tensor(attention_mask),
            torch.tensor(loss_mask),
        )

    def collate_fn(self, batch):
        input_ids, attention_mask, loss_mask = zip(*batch)

        input_ids = zero_pad_sequences(
            input_ids, value=self.tokenizer.pad_token_id, side="right"
        )
        attention_mask = zero_pad_sequences(attention_mask, value=0, side="right")
        loss_mask = zero_pad_sequences(loss_mask, value=0, side="right")
        return input_ids, attention_mask, loss_mask
