#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：DeepSeek 
@File    ：inference_training.py
@Author  ：Neil
@Date    ：2025/2/13 18:09 
"""

#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from swift.llm import (
    get_model_tokenizer,
    load_dataset,
    get_template,
    EncodePreprocessor,
    ModelType
)
from swift.utils import (
    get_logger,
    read_multi_line,
    get_model_parameter_info,
    plot_images,
    seed_everything,
    find_all_linears
)
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from functools import partial
logger = get_logger()
seed_everything(42)

def main(output_dir):
    # training_args
    global training_args
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_checkpointing=True,
        weight_decay=0.1,
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        report_to=['tensorboard'],
        logging_first_step=True,
        save_strategy='steps',
        save_steps=50,
        eval_strategy='steps',
        eval_steps=50,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        metric_for_best_model='loss',
        save_total_limit=5,
        logging_steps=5,
        dataloader_num_workers=1,
        data_seed=data_seed,
    )
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    logger.info(f'output_dir: {output_dir}')


    # Obtain the model and template, and add a trainable Lora layer on the model.
    model, tokenizer = get_model_tokenizer(model_type=ModelType.deepseek_r1_distill, model_id_or_path=model_id_or_path,)
    logger.info(f'model_info: {model.model_info}')
    template = get_template(model.model_meta.template, tokenizer, default_system=system, max_length=max_length)
    template.set_mode('train')

    target_modules = find_all_linears(model)
    # print(target_modules) # ['down_proj', 'k_proj', 'up_proj', 'v_proj', 'o_proj', 'gate_proj', 'q_proj']
    lora_config = LoraConfig(task_type='CAUSAL_LM', r=lora_rank, lora_alpha=lora_alpha,
                             target_modules=target_modules)
    model = Swift.prepare_model(model, lora_config)
    logger.info(f'lora_config: {lora_config}')

    # Print model structure and trainable parameters.
    logger.info(f'model: {model}')
    model_parameter_info = get_model_parameter_info(model)
    logger.info(f'model_parameter_info: {model_parameter_info}')

    # Download and load the dataset, split it into a training set and a validation set,
    # and encode the text data into tokens.
    print(123123)
    train_dataset, val_dataset = load_dataset(dataset, split_dataset_ratio=split_dataset_ratio, num_proc=num_proc,
            model_name=model_name, model_author=model_author, seed=data_seed)
    print(456456)
    logger.info(f'train_dataset: {train_dataset}')
    logger.info(f'val_dataset: {val_dataset}')
    logger.info(f'train_dataset[0]: {train_dataset[0]}')

    train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=num_proc)
    val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=num_proc)
    logger.info(f'encoded_train_dataset[0]: {train_dataset[0]}')

    # Print a sample
    template.print_inputs(train_dataset[0])
    print(567567567)
    # Get the trainer and start the training.
    model.enable_input_require_grads()  # Compatible with gradient checkpointing
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=template.data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        template=template,
    )
    print(123123123123)
    trainer.train()
    print(14144114141414)
    last_model_checkpoint = trainer.state.last_model_checkpoint
    logger.info(f'last_model_checkpoint: {last_model_checkpoint}')

def results_demo():
    # Visualize the training loss.
    # You can also use the TensorBoard visualization interface during training by entering
    # `tensorboard --logdir '{output_dir}/runs'` at the command line.
    images_dir = os.path.join(output_dir, 'images')
    logger.info(f'images_dir: {images_dir}')
    plot_images(images_dir, training_args.logging_dir, ['train/loss'], 0.9)  # save images

    # Read and display the image.
    # The light yellow line represents the actual loss value,
    # while the yellow line represents the loss value smoothed with a smoothing factor of 0.9.
    # from IPython.display import display

    from PIL import Image
    import matplotlib.pyplot as plt
    image = Image.open(os.path.join(images_dir, 'train_loss.png'))
    plt.figure("Train_loss")
    plt.imshow(image)
    plt.axis("on")
    plt.title("Train loss")
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    model_id_or_path = "Model/R1_Distil"
    # model_id_or_path = "Qwen/Qwen2.5-3B-Instruct"
    system = 'You are a helpful psychological consultant.'
    output_dir = 'Output'
    dataset = [
        'AI-ModelScope/alpaca-gpt4-data-zh',
        # 'AI-ModelScope/alpaca-gpt4-data-en',
        'swift/self-cognition'
    ]
    data_seed = 42
    max_length = 1024
    split_dataset_ratio = 0.01  # Split validation set
    num_proc = 1  # The number of processes for data loading.
    # The following two parameters are used to override the placeholders in the self-cognition dataset.
    model_name = ['小心心', 'Sweet Heart']  # The Chinese name and English name of the model
    model_author = ['张小屁', 'Neil']  # The Chinese name

    # lora
    lora_rank = 8
    lora_alpha = 32

    main(output_dir)
    results_demo()














# from swift.llm import DatasetName, ModelType, SftArguments, sft_main
#
# sft_args = SftArguments(
#     model_type=ModelType.deepseek_r1_distill,
#     model_id_or_path= "Model/R1_Distill",
#     dataset=[f'{DatasetName.alpaca_zh}#500',
#              f'{DatasetName.self_cognition}#500',
#     ],
#     max_length=1024,
#     learning_rate=1e-4,
#     output_dir='output',
#     lora_target_modules=["attn.c_attn", "attn.c_proj", "mlp.w1", "mlp.w2", "mlp.c_proj"],
#     model_name=['小心心'],
#     model_author=['Neil'],
#     device_map_config='0')
# output = sft_main(sft_args)
# best_model_checkpoint = output['best_model_checkpoint']
# print(f'best_model_checkpoint: {best_model_checkpoint}')


