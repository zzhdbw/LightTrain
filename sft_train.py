import csv
from dataclasses import asdict

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dataset import SFTDataSet
from src.sft_trainer import SFTTrainer, TrainArgs


def load_data(data_path: str) -> list[list[dict]]:
    # 加载data_path
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_data = [row for row in reader]

    data = [
        [
            {"role": "system", "content": "你是一个有用的助手"},
            {
                "role": "user",
                "content": (
                    item["instruction"] + "\n" + item["input"]
                    if item["input"]
                    else item["instruction"]
                ),
            },
            {"role": "assistant", "content": item["output"]},
        ]
        for item in raw_data
    ]
    return data


def load_self_identify_data(data_path: str) -> list[list[dict]]:
    # 加载data_path
    import json

    raw_data = json.load(open(data_path, "r", encoding="utf-8"))

    data = [
        [
            {"role": "system", "content": "你是一个有用的助手"},
            {
                "role": "user",
                "content": (
                    item["instruction"] + "\n" + item["input"]
                    if item["input"]
                    else item["instruction"]
                ),
            },
            {"role": "assistant", "content": item["output"]},
        ]
        for item in raw_data
    ]
    return data


def get_args():
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--data_path", type=str, required=True)
    args.add_argument("--model_path", type=str, required=True)
    args.add_argument("--output_dir", type=str, default="/mnt/afs/zzh/ckpt")
    args.add_argument("--num_train_epochs", type=int, default=8)
    args.add_argument("--learning_rate", type=float, default=2e-5)
    args.add_argument("--per_device_train_batch_size", type=int, default=2)
    args.add_argument("--per_device_eval_batch_size", type=int, default=2)
    args.add_argument("--gradient_accumulation_steps", type=int, default=4)
    args.add_argument("--eval_steps", type=int, default=100)
    args.add_argument("--save_steps", type=int, default=200)
    args.add_argument("--save_total_limit", type=int, default=2)
    args.add_argument("--logging_steps", type=int, default=2)
    args.add_argument("--lr_scheduler_name", type=str, default="")
    args.add_argument("--lr_scheduler_num_warmup_steps", type=int, default=10)
    args.add_argument("--lr_min", type=float, default=1e-7)
    args.add_argument("--dataloader_num_workers", type=int, default=4)
    args.add_argument("--train_on_prompt", action="store_true")
    args.add_argument("--use_dft_loss", action="store_true")
    args.add_argument("--dft_alpha", type=float, default=0.8)
    args.add_argument("--report_to", type=str, default="swanlab")
    args.add_argument("--swanlab_project_name", type=str, default="LightLLMTrainer")
    args.add_argument("--swanlab_group_name", type=str, default="SFT Training")
    args.add_argument("--gradient_checkpointing", action="store_true")
    return args.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.data_path.endswith(".json"):
        data = load_self_identify_data(args.data_path)
    else:
        data = load_data(args.data_path)

    model = AutoModelForCausalLM.from_pretrained(args.model_path).to("cuda")

    if args.gradient_checkpointing:
        # 开启梯度检查点
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    train_dataset = SFTDataSet(
        data, num_proc=1, tokenizer=tokenizer, train_on_prompt=args.train_on_prompt
    )

    train_args = TrainArgs(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        lr_scheduler_name=args.lr_scheduler_name,
        lr_scheduler_num_warmup_steps=args.lr_scheduler_num_warmup_steps,
        lr_min=args.lr_min,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=args.report_to,
        use_dft_loss=args.use_dft_loss,
        dft_alpha=args.dft_alpha,
        swanlab_project_name=args.swanlab_project_name,
        swanlab_group_name=args.swanlab_group_name,
    )

    # 打印train_args
    print(asdict(train_args))

    trainer = SFTTrainer(
        model=model,
        train_args=train_args,
        train_dataset=train_dataset,
    )
    trainer.fit()
