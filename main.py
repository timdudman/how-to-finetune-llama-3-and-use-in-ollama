from unsloth import FastLanguageModel
from datasets import load_dataset
from unsloth import to_sharegpt
from pprint import pprint
from unsloth import standardize_sharegpt
from unsloth import apply_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import torch
import subprocess
import ollama

max_seq_length = 2048
dtype = None  # Auto-detect
load_in_4bit = True  # Use 4bit quantization to reduce memory usage


def main():
    # Use Llama 3.2 with 1 billion parameters, binary neural network and 4-bit quantisation
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3.2-1b-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Add LoRA adapters so we only need to update 1 to 10% of all parameters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Rank of the finetuning process
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Train on all modules
        lora_alpha=16,  # Scaling factor for finetuning (set to equal rank)
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # type: ignore # Unsloth here reduces memory usage
        random_state=3407,  # Random seed
        use_rslora=False,  # Would use lora_alpha = 16 auromatically
        loftq_config=None,  # Initialisation of LoRA matrices
    )

    # Load the Titanic dataset (https://huggingface.co/datasets/unsloth/datasets)
    dataset = load_dataset(
        "csv",
        data_files="https://huggingface.co/datasets/unsloth/datasets/raw/main/titanic.csv",
        split="train",
    )

    # Example raw data
    pprint(dataset[0])  # type: ignore

    # Merge columns to predict survived (or not)
    dataset = to_sharegpt(
        dataset,
        merged_prompt="[[The passenger embarked from {Embarked}.]]"  # [[]] is for optional values (if they are missing)
        "[[\nThey are {Sex}.]]"
        "[[\nThey have {Parch} parents and childen.]]"
        "[[\nThey have {SibSp} siblings and spouses.]]"
        "[[\nTheir passenger class is {Pclass}.]]"
        "[[\nTheir age is {Age}.]]"
        "[[\nThey paid ${Fare} for the trip.]]",
        conversation_extension=5,  # Randomnly combines conversations into 1! Good for long convos
        output_column_name="Survived",
    )

    # Standardise into "user" and "assistant" tags
    dataset = standardize_sharegpt(dataset)

    # Apply a chat template
    chat_template = """Below describes some details about some passengers who went on the Titanic.
    Predict whether they survived or perished based on their characteristics.
    Output 1 if they survived, and 0 if they died.
    >>> Passenger Details:
    {INPUT}
    >>> Did they survive?
    {OUTPUT}"""

    dataset = apply_chat_template(
        dataset,
        tokenizer=tokenizer,
        chat_template=chat_template,
        # default_system_message = "You are a helpful assistant", << [OPTIONAL]
    )

    # Example training data
    pprint(dataset[0])

    # Train the model
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,  # type: ignore
        train_dataset=dataset,
        dataset_text_field="text",  # type: ignore
        max_seq_length=max_seq_length,  # type: ignore
        dataset_num_proc=2,  # type: ignore
        packing=False,  # type: ignore Can make training 5x faster for short sequences
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=1,  # max_steps=60 to make it faster
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )

    # Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    pprint(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    pprint(f"{start_gpu_memory} GB of memory reserved.")

    # Train
    trainer_stats = trainer.train()

    # Show final memory stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    pprint(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    pprint(
        f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
    )
    pprint(f"Peak reserved memory = {used_memory} GB.")
    pprint(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    pprint(f"Peak reserved memory % of max memory = {used_percentage} %.")
    pprint(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # Save GGUF for Ollama
    model.save_pretrained_gguf(
        "model",
        tokenizer,
    )

    pprint(tokenizer._ollama_modelfile)

    # Create Ollama model
    subprocess.run(["ollama", "create", "unsloth_model", "-f", "./model/Modelfile"])

    # Perform inference using Ollama!
    response = ollama.chat(
        model="unsloth_model",
        messages=[
            {
                "role": "user",
                "content": "Their passenger class is 3.\nTheir age is 22.0.\nThey paid $107.25 for the trip.",
            }
        ],
    )
    pprint(response)


if __name__ == "__main__":
    main()
