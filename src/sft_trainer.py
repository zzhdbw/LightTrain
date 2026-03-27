from abc import ABC
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

from .dataset import SFTDataSet


@dataclass
class TrainArgs:
    output_dir: str
    num_train_epochs: int
    learning_rate: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    eval_steps: int
    save_steps: int
    save_total_limit: int
    logging_steps: int
    lr_scheduler_type: str
    dataloader_num_workers: int
    optim_beta: tuple = (0.9, 0.999)
    optim_eps: float = 1e-6
    optim_weight_decay: float = 0.01
    lr_scheduler_name: str = "cosine"
    lr_scheduler_num_warmup_steps: int = 10
    report_to: str = None
    swanlab_project_name: str = "LightLLMTrainer"
    swanlab_group_name: str = "SFT Training"


class SFTTrainer(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        train_args: TrainArgs,
        train_dataset: SFTDataSet,
        eval_dataset: SFTDataSet = None,
    ):

        self.model = model
        self.tokenizer = train_dataset.tokenizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_args.learning_rate,
            betas=train_args.optim_beta,
            eps=train_args.optim_eps,
            weight_decay=train_args.optim_weight_decay,
            foreach=True,
        )
        self.train_dataset = train_dataset
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=train_args.per_device_train_batch_size,
            shuffle=True,
            num_workers=train_args.dataloader_num_workers,
            collate_fn=self.train_dataset.collate_fn,
        )
        self.eval_dataset = eval_dataset
        if self.eval_dataset:
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=train_args.per_device_eval_batch_size,
                shuffle=False,
                num_workers=train_args.dataloader_num_workers,
                collate_fn=self.eval_dataset.collate_fn,
            )
        self.train_args = train_args
        self.num_train_epochs = train_args.num_train_epochs

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        self.gradient_accumulation_steps = train_args.gradient_accumulation_steps
        self.num_training_steps = (
            len(self.train_dataloader)
            * self.num_train_epochs
            // self.gradient_accumulation_steps
        ) + 1
        self.lr_scheduler = get_scheduler(
            name=train_args.lr_scheduler_name,  # 可改成 "linear", "constant_with_warmup" 等
            optimizer=self.optimizer,
            num_warmup_steps=train_args.lr_scheduler_num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )

        self.report_to = train_args.report_to
        if self.report_to:
            assert self.report_to in ["swanlab"], "report_to only support swanlab now !"
        if self.report_to == "swanlab":
            import swanlab

            swanlab.init(
                project=train_args.swanlab_project_name,
                experiment_name=train_args.swanlab_group_name,
                config=asdict(train_args),
            )
            self._logger = swanlab

    def fit(self) -> None:

        step = 1  # for logging and gradient_accumulate
        global_step = 1  # actual step

        for epoch in range(1, self.num_train_epochs + 1):
            self.model.train()

            for _, (input_ids, attention_mask, loss_mask) in enumerate(
                self.train_dataloader
            ):
                input_ids, attention_mask, loss_mask = (
                    input_ids.to("cuda"),
                    attention_mask.to("cuda"),
                    loss_mask.to("cuda"),
                )

                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits

                loss = self.calculate_loss(logits, input_ids, loss_mask)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                # 梯度累积
                if step % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                # 日志
                if step % self.train_args.logging_steps == 0:
                    self.log(
                        epoch=epoch,
                        global_step=global_step,
                        loss=loss.item(),
                        lr=self.lr_scheduler.get_last_lr()[0],
                    )

                # 评估
                if step % self.train_args.eval_steps == 0 and self.eval_dataset:
                    self.evaluate(epoch=epoch, global_step=global_step)

                step += 1

        self.save_checkpoints(global_step)

    def evaluate(self, epoch: int, global_step: int) -> None:
        self.model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for input_ids, attention_mask, loss_mask in self.eval_dataloader:
                input_ids, attention_mask, loss_mask = (
                    input_ids.to("cuda"),
                    attention_mask.to("cuda"),
                    loss_mask.to("cuda"),
                )
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
                loss = self.calculate_loss(logits, input_ids, loss_mask)
                eval_loss += loss.item()

        avg_eval_loss = eval_loss / len(self.eval_dataloader)
        print(
            f"Epoch {epoch} | Step: {global_step}/{self.num_training_steps} | Evaluation loss: {avg_eval_loss:.4f}"
        )

        if self.report_to == "swanlab":
            self._logger.log({"eval/loss": avg_eval_loss}, step=global_step)

    def save_checkpoints(self, step: int) -> None:

        output_path = Path(self.train_args.output_dir)
        if step is not None:
            output_path = output_path / str(step)

        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

    def calculate_loss(
        self, logits: torch.Tensor, input_ids: torch.Tensor, loss_mask: torch.Tensor
    ) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_loss_mask = loss_mask[:, 1:].contiguous()

        loss = self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        loss = loss.view(shift_labels.size())
        loss = (loss * shift_loss_mask).sum() / shift_loss_mask.sum()
        return loss

    def log(self, epoch: int, global_step: int, loss: float, lr: float) -> None:
        print(
            f"Epoch {epoch} | Step: {global_step}/{self.num_training_steps} | Loss: {loss * self.gradient_accumulation_steps:.4f} | LR: {lr:.2e}"
        )
        if self.report_to == "swanlab":
            self._logger.log(
                {
                    "epoch": epoch,
                    "train/loss": loss * self.gradient_accumulation_steps,
                    "train/learning_rate": lr,
                },
                step=global_step,
            )
