#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation.py

usage: train.py [-h] --model_name_or_path MODEL_NAME_OR_PATH [--config_name CONFIG_NAME]                
                [--tokenizer_name TOKENIZER_NAME] [--cache_dir CACHE_DIR]                               
                [--use_fast_tokenizer [USE_FAST_TOKENIZER]] [--no_use_fast_tokenizer]                    
                [--model_revision MODEL_REVISION] [--token TOKEN] [--use_auth_token [USE_AUTH_TOKEN]]   
                [--trust_remote_code [TRUST_REMOTE_CODE]] --dataset_path DATASET_PATH                   
                [--metric_name_or_path METRIC_NAME_OR_PATH] [--source_lang SOURCE_LANG]              
                [--target_lang TARGET_LANG] [--overwrite_cache [OVERWRITE_CACHE]]                       
                [--preprocessing_num_workers PREPROCESSING_NUM_WORKERS]                                 
                [--max_source_length MAX_SOURCE_LENGTH] [--pad_to_max_length [PAD_TO_MAX_LENGTH]]       
                [--max_train_samples MAX_TRAIN_SAMPLES] [--max_eval_samples MAX_EVAL_SAMPLES]           
                [--max_predict_samples MAX_PREDICT_SAMPLES]                                             
                [--ignore_pad_token_for_loss [IGNORE_PAD_TOKEN_FOR_LOSS]]                               
                [--no_ignore_pad_token_for_loss] [--source_prefix SOURCE_PREFIX]                        
                [--forced_bos_token FORCED_BOS_TOKEN] --output_dir OUTPUT_DIR                           
                [--overwrite_output_dir [OVERWRITE_OUTPUT_DIR]] [--do_train [DO_TRAIN]]                 
                [--do_eval [DO_EVAL]] [--do_predict [DO_PREDICT]]                                      
                [--evaluation_strategy {no,steps,epoch}]                                                
                [--prediction_loss_only [PREDICTION_LOSS_ONLY]]                                         
                [--per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE]                             
                [--per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE]                               
                [--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE]
                [--per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE]
                [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                [--eval_accumulation_steps EVAL_ACCUMULATION_STEPS] [--eval_delay EVAL_DELAY]
                [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY] [--adam_beta1 ADAM_BETA1]
                [--adam_beta2 ADAM_BETA2] [--adam_epsilon ADAM_EPSILON] [--max_grad_norm MAX_GRAD_NORM]
                [--num_train_epochs NUM_TRAIN_EPOCHS] [--max_steps MAX_STEPS]
                [--lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
                [--warmup_ratio WARMUP_RATIO] [--warmup_steps WARMUP_STEPS]
                [--log_level {debug,info,warning,error,critical,passive}]
                [--log_level_replica {debug,info,warning,error,critical,passive}]
                [--log_on_each_node [LOG_ON_EACH_NODE]] [--no_log_on_each_node]
                [--logging_dir LOGGING_DIR] [--logging_strategy {no,steps,epoch}]
                [--logging_first_step [LOGGING_FIRST_STEP]] [--logging_steps LOGGING_STEPS]
                [--logging_nan_inf_filter [LOGGING_NAN_INF_FILTER]] [--no_logging_nan_inf_filter]
                [--save_strategy {no,steps,epoch}] [--save_steps SAVE_STEPS]
                [--save_total_limit SAVE_TOTAL_LIMIT] [--save_on_each_node [SAVE_ON_EACH_NODE]]
                [--no_cuda [NO_CUDA]] [--use_mps_device [USE_MPS_DEVICE]] [--seed SEED]
                [--data_seed DATA_SEED] [--jit_mode_eval [JIT_MODE_EVAL]] [--use_ipex [USE_IPEX]]
                [--bf16 [BF16]] [--fp16 [FP16]] [--fp16_opt_level FP16_OPT_LEVEL]
                [--half_precision_backend {auto,cuda_amp,apex,cpu_amp}]
                [--bf16_full_eval [BF16_FULL_EVAL]] [--fp16_full_eval [FP16_FULL_EVAL]] [--tf32 TF32]
                [--local_rank LOCAL_RANK] [--xpu_backend {mpi,ccl}] [--tpu_num_cores TPU_NUM_CORES]
                [--tpu_metrics_debug [TPU_METRICS_DEBUG]] [--debug DEBUG]
                [--dataloader_drop_last [DATALOADER_DROP_LAST]] [--eval_steps EVAL_STEPS]
                [--dataloader_num_workers DATALOADER_NUM_WORKERS] [--past_index PAST_INDEX]
                [--run_name RUN_NAME] [--disable_tqdm DISABLE_TQDM]
                [--remove_unused_columns [REMOVE_UNUSED_COLUMNS]] [--no_remove_unused_columns]
                [--label_names LABEL_NAMES [LABEL_NAMES ...]]
                [--load_best_model_at_end [LOAD_BEST_MODEL_AT_END]]
                [--metric_for_best_model METRIC_FOR_BEST_MODEL] [--greater_is_better GREATER_IS_BETTER]
                [--ignore_data_skip [IGNORE_DATA_SKIP]] [--sharded_ddp SHARDED_DDP] [--fsdp FSDP]
                [--fsdp_min_num_params FSDP_MIN_NUM_PARAMS]
                [--fsdp_transformer_layer_cls_to_wrap FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP]
                [--deepspeed DEEPSPEED] [--label_smoothing_factor LABEL_SMOOTHING_FACTOR]
                [--optim {adamw_hf,adamw_torch,adamw_torch_xla,adamw_apex_fused,adafactor,adamw_bnb_8bit,sgd,adagrad}]
                [--adafactor [ADAFACTOR]] [--group_by_length [GROUP_BY_LENGTH]]
                [--length_column_name LENGTH_COLUMN_NAME] [--report_to REPORT_TO [REPORT_TO ...]]
                [--ddp_find_unused_parameters DDP_FIND_UNUSED_PARAMETERS]
                [--ddp_bucket_cap_mb DDP_BUCKET_CAP_MB]
                [--dataloader_pin_memory [DATALOADER_PIN_MEMORY]] [--no_dataloader_pin_memory]
                [--skip_memory_metrics [SKIP_MEMORY_METRICS]] [--no_skip_memory_metrics]
                [--use_legacy_prediction_loop [USE_LEGACY_PREDICTION_LOOP]]
                [--push_to_hub [PUSH_TO_HUB]] [--resume_from_checkpoint RESUME_FROM_CHECKPOINT]
                [--hub_model_id HUB_MODEL_ID]
                [--hub_strategy {end,every_save,checkpoint,all_checkpoints}] [--hub_token HUB_TOKEN]
                [--hub_private_repo [HUB_PRIVATE_REPO]]
                [--gradient_checkpointing [GRADIENT_CHECKPOINTING]]
                [--include_inputs_for_metrics [INCLUDE_INPUTS_FOR_METRICS]]
                [--fp16_backend {auto,cuda_amp,apex,cpu_amp}]
                [--push_to_hub_model_id PUSH_TO_HUB_MODEL_ID]
                [--push_to_hub_organization PUSH_TO_HUB_ORGANIZATION]
                [--push_to_hub_token PUSH_TO_HUB_TOKEN] [--mp_parameters MP_PARAMETERS]
                [--auto_find_batch_size [AUTO_FIND_BATCH_SIZE]] [--full_determinism [FULL_DETERMINISM]]
                [--torchdynamo {eager,nvfuser,fx2trt,fx2trt-fp16}] [--ray_scope RAY_SCOPE]
                [--ddp_timeout DDP_TIMEOUT] [--sortish_sampler [SORTISH_SAMPLER]]
                [--predict_with_generate [PREDICT_WITH_GENERATE]]
                [--generation_max_length GENERATION_MAX_LENGTH]
                [--generation_num_beams GENERATION_NUM_BEAMS]

options:
  -h, --help            show this help message and exit
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to pretrained model or model identifier from huggingface.co/models
                        (default: None)
  --config_name CONFIG_NAME
                        Pretrained config name or path if not the same as model_name (default: None)
  --tokenizer_name TOKENIZER_NAME
                        Pretrained tokenizer name or path if not the same as model_name (default: None)
  --cache_dir CACHE_DIR
                        Where to store the pretrained models downloaded from huggingface.co (default:
                        None)
  --use_fast_tokenizer [USE_FAST_TOKENIZER]
                        Whether to use one of the fast tokenizer (backed by the tokenizers library) or
                        not. (default: True)
  --no_use_fast_tokenizer
                        Whether to use one of the fast tokenizer (backed by the tokenizers library) or
                        not. (default: False)
  --model_revision MODEL_REVISION
                        The specific model version to use (can be a branch name, tag name or commit
                        id). (default: main)
  --token TOKEN         The token to use as HTTP bearer authorization for remote files. If not
                        specified, will use the token generated when running `huggingface-cli login`
                        (stored in `~/.huggingface`). (default: None)
  --use_auth_token [USE_AUTH_TOKEN]
                        The `use_auth_token` argument is deprecated and will be removed in v4.34.
                        Please use `token`. (default: None)
  --trust_remote_code [TRUST_REMOTE_CODE]
                        Whether or not to allow for custom models defined on the Hub in their own
                        modeling files. This optionshould only be set to `True` for repositories you
                        trust and in which you have read the code, as it willexecute code present on
                        the Hub on your local machine. (default: False)
  --dataset_path DATASET_PATH
                        Path to the dataset to load from disk (via the datasets library). (default:
                        None)
  --metric_name_or_path METRIC_NAME_OR_PATH
  --source_lang SOURCE_LANG
                        Source language id for translation. (default: None)
  --target_lang TARGET_LANG
                        Target language id for translation. (default: None)
  --overwrite_cache [OVERWRITE_CACHE]
                        Overwrite the cached training and evaluation sets (default: False)
  --preprocessing_num_workers PREPROCESSING_NUM_WORKERS
                        The number of processes to use for the preprocessing. (default: None)
  --max_source_length MAX_SOURCE_LENGTH
                        The maximum total input sequence length after tokenization. Sequences longer
                        than this will be truncated, sequences shorter will be padded. (default: 1024)
  --pad_to_max_length [PAD_TO_MAX_LENGTH]
                        Whether to pad all samples to model maximum sentence length. If False, will pad
                        the samples dynamically when batching to the maximum length in the batch. More
                        efficient on GPU but very bad for TPU. (default: False)
  --max_train_samples MAX_TRAIN_SAMPLES
                        For debugging purposes or quicker training, truncate the number of training
                        examples to this value if set. (default: None)
  --max_eval_samples MAX_EVAL_SAMPLES
                        For debugging purposes or quicker training, truncate the number of evaluation
                        examples to this value if set. (default: None)
  --max_predict_samples MAX_PREDICT_SAMPLES
                        For debugging purposes or quicker training, truncate the number of prediction
                        examples to this value if set. (default: None)
  --ignore_pad_token_for_loss [IGNORE_PAD_TOKEN_FOR_LOSS]
                        Whether to ignore the tokens corresponding to padded labels in the loss
                        computation or not. (default: True)
  --no_ignore_pad_token_for_loss
                        Whether to ignore the tokens corresponding to padded labels in the loss
                        computation or not. (default: False)
  --source_prefix SOURCE_PREFIX
                        A prefix to add before every source text (useful for T5 models). (default:
                        None)
  --forced_bos_token FORCED_BOS_TOKEN
                        The token to force as the first generated token after the
                        :obj:`decoder_start_token_id`.Useful for multilingual models like :doc:`mBART
                        <../model_doc/mbart>` where the first generated token needs to be the target
                        language token.(Usually it is the target language token) (default: None)
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and checkpoints will be
                        written. (default: None)
  --overwrite_output_dir [OVERWRITE_OUTPUT_DIR]
                        Overwrite the content of the output directory. Use this to continue training if
                        output_dir points to a checkpoint directory. (default: False)
  --do_train [DO_TRAIN]
                        Whether to run training. (default: False)
  --do_eval [DO_EVAL]   Whether to run eval on the dev set. (default: False)
  --do_predict [DO_PREDICT]
                        Whether to run predictions on the test set. (default: False)
  --evaluation_strategy {no,steps,epoch}
                        The evaluation strategy to use. (default: no)
  --prediction_loss_only [PREDICTION_LOSS_ONLY]
                        When performing evaluation and predictions, only returns the loss. (default:
                        False)
  --per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE
                        Batch size per GPU/TPU core/CPU for training. (default: 8)
  --per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE
                        Batch size per GPU/TPU core/CPU for evaluation. (default: 8)
  --per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE
                        Deprecated, the use of `--per_device_train_batch_size` is preferred. Batch size
                        per GPU/TPU core/CPU for training. (default: None)
  --per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE
                        Deprecated, the use of `--per_device_eval_batch_size` is preferred. Batch size
                        per GPU/TPU core/CPU for evaluation. (default: None)
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before performing a backward/update pass.
                        (default: 1)
  --eval_accumulation_steps EVAL_ACCUMULATION_STEPS
                        Number of predictions steps to accumulate before moving the tensors to the CPU.
                        (default: None)
  --eval_delay EVAL_DELAY
                        Number of epochs or steps to wait for before the first evaluation can be
                        performed, depending on the evaluation_strategy. (default: 0)
  --learning_rate LEARNING_RATE
                        The initial learning rate for AdamW. (default: 5e-05)
  --weight_decay WEIGHT_DECAY
                        Weight decay for AdamW if we apply some. (default: 0.0)
  --adam_beta1 ADAM_BETA1
                        Beta1 for AdamW optimizer (default: 0.9)
  --adam_beta2 ADAM_BETA2
                        Beta2 for AdamW optimizer (default: 0.999)
  --adam_epsilon ADAM_EPSILON
                        Epsilon for AdamW optimizer. (default: 1e-08)
  --max_grad_norm MAX_GRAD_NORM
                        Max gradient norm. (default: 1.0)
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform. (default: 3.0)
  --max_steps MAX_STEPS
                        If > 0: set total number of training steps to perform. Override
                        num_train_epochs. (default: -1)
  --lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}
                        The scheduler type to use. (default: linear)
  --warmup_ratio WARMUP_RATIO
                        Linear warmup over warmup_ratio fraction of total steps. (default: 0.0)
  --warmup_steps WARMUP_STEPS
                        Linear warmup over warmup_steps. (default: 0)
  --log_level {debug,info,warning,error,critical,passive}
                        Logger log level to use on the main node. Possible choices are the log levels
                        as strings: 'debug', 'info', 'warning', 'error' and 'critical', plus a
                        'passive' level which doesn't set anything and lets the application set the
                        level. Defaults to 'passive'. (default: passive)
  --log_level_replica {debug,info,warning,error,critical,passive}
                        Logger log level to use on replica nodes. Same choices and defaults as
                        ``log_level`` (default: passive)
  --log_on_each_node [LOG_ON_EACH_NODE]
                        When doing a multinode distributed training, whether to log once per node or
                        just once on the main node. (default: True)
  --no_log_on_each_node
                        When doing a multinode distributed training, whether to log once per node or
                        just once on the main node. (default: False)
  --logging_dir LOGGING_DIR
                        Tensorboard log dir. (default: None)
  --logging_strategy {no,steps,epoch}
                        The logging strategy to use. (default: steps)
  --logging_first_step [LOGGING_FIRST_STEP]
                        Log the first global_step (default: False)
  --logging_steps LOGGING_STEPS
                        Log every X updates steps. (default: 500)
  --logging_nan_inf_filter [LOGGING_NAN_INF_FILTER]
                        Filter nan and inf losses for logging. (default: True)
  --no_logging_nan_inf_filter
                        Filter nan and inf losses for logging. (default: False)
  --save_strategy {no,steps,epoch}
                        The checkpoint save strategy to use. (default: steps)
  --save_steps SAVE_STEPS
                        Save checkpoint every X updates steps. (default: 500)
  --save_total_limit SAVE_TOTAL_LIMIT
                        Limit the total amount of checkpoints. Deletes the older checkpoints in the
                        output_dir. Default is unlimited checkpoints (default: None)
  --save_on_each_node [SAVE_ON_EACH_NODE]
                        When doing multi-node distributed training, whether to save models and
                        checkpoints on each node, or only on the main one (default: False)
  --no_cuda [NO_CUDA]   Do not use CUDA even when it is available (default: False)
  --use_mps_device [USE_MPS_DEVICE]
                        Whether to use Apple Silicon chip based `mps` device. (default: False)
  --seed SEED           Random seed that will be set at the beginning of training. (default: 42)
  --data_seed DATA_SEED
                        Random seed to be used with data samplers. (default: None)
  --jit_mode_eval [JIT_MODE_EVAL]
                        Whether or not to use PyTorch jit trace for inference (default: False)
  --use_ipex [USE_IPEX]
                        Use Intel extension for PyTorch when it is available, installation:
                        'https://github.com/intel/intel-extension-for-pytorch' (default: False)
  --bf16 [BF16]         Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or
                        higher NVIDIA architecture or using CPU (no_cuda). This is an experimental API
                        and it may change. (default: False)
  --fp16 [FP16]         Whether to use fp16 (mixed) precision instead of 32-bit (default: False)
  --fp16_opt_level FP16_OPT_LEVEL
                        For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].
                        See details at https://nvidia.github.io/apex/amp.html (default: O1)
  --half_precision_backend {auto,cuda_amp,apex,cpu_amp}
                        The backend to be used for half precision. (default: auto)
  --bf16_full_eval [BF16_FULL_EVAL]
                        Whether to use full bfloat16 evaluation instead of 32-bit. This is an
                        experimental API and it may change. (default: False)
  --fp16_full_eval [FP16_FULL_EVAL]
                        Whether to use full float16 evaluation instead of 32-bit (default: False)
  --tf32 TF32           Whether to enable tf32 mode, available in Ampere and newer GPU architectures.
                        This is an experimental API and it may change. (default: None)
  --local_rank LOCAL_RANK
                        For distributed training: local_rank (default: -1)
  --xpu_backend {mpi,ccl}
                        The backend to be used for distributed training on Intel XPU. (default: None)
  --tpu_num_cores TPU_NUM_CORES
                        TPU: Number of TPU cores (automatically passed by launcher script) (default:
                        None)
  --tpu_metrics_debug [TPU_METRICS_DEBUG]
                        Deprecated, the use of `--debug tpu_metrics_debug` is preferred. TPU: Whether
                        to print debug metrics (default: False)
  --debug DEBUG         Whether or not to enable debug mode. Current options: `underflow_overflow`
                        (Detect underflow and overflow in activations and weights), `tpu_metrics_debug`
                        (print debug metrics on TPU). (default: )
  --dataloader_drop_last [DATALOADER_DROP_LAST]
                        Drop the last incomplete batch if it is not divisible by the batch size.
                        (default: False)
  --eval_steps EVAL_STEPS
                        Run an evaluation every X steps. (default: None)
  --dataloader_num_workers DATALOADER_NUM_WORKERS
                        Number of subprocesses to use for data loading (PyTorch only). 0 means that the
                        data will be loaded in the main process. (default: 0)
  --past_index PAST_INDEX
                        If >=0, uses the corresponding part of the output as the past state for next
                        step. (default: -1)
  --run_name RUN_NAME   An optional descriptor for the run. Notably used for wandb logging. (default:
                        None)
  --disable_tqdm DISABLE_TQDM
                        Whether or not to disable the tqdm progress bars. (default: None)
  --remove_unused_columns [REMOVE_UNUSED_COLUMNS]
                        Remove columns not required by the model when using an nlp.Dataset. (default:
                        True)
  --no_remove_unused_columns
                        Remove columns not required by the model when using an nlp.Dataset. (default:
                        False)
  --label_names LABEL_NAMES [LABEL_NAMES ...]
                        The list of keys in your dictionary of inputs that correspond to the labels.
                        (default: None)
  --load_best_model_at_end [LOAD_BEST_MODEL_AT_END]
                        Whether or not to load the best model found during training at the end of
                        training. (default: False)
  --metric_for_best_model METRIC_FOR_BEST_MODEL
                        The metric to use to compare two different models. (default: None)
  --greater_is_better GREATER_IS_BETTER
                        Whether the `metric_for_best_model` should be maximized or not. (default: None)
  --ignore_data_skip [IGNORE_DATA_SKIP]
                        When resuming training, whether or not to skip the first epochs and batches to
                        get to the same training data. (default: False)
  --sharded_ddp SHARDED_DDP
                        Whether or not to use sharded DDP training (in distributed training only). The
                        base option should be `simple`, `zero_dp_2` or `zero_dp_3` and you can add CPU-
                        offload to `zero_dp_2` or `zero_dp_3` like this: zero_dp_2 offload` or
                        `zero_dp_3 offload`. You can add auto-wrap to `zero_dp_2` or `zero_dp_3` with
                        the same syntax: zero_dp_2 auto_wrap` or `zero_dp_3 auto_wrap`. (default: )
  --fsdp FSDP           Whether or not to use PyTorch Fully Sharded Data Parallel (FSDP) training (in
                        distributed training only). The base option should be `full_shard`,
                        `shard_grad_op` or `no_shard` and you can add CPU-offload to `full_shard` or
                        `shard_grad_op` like this: full_shard offload` or `shard_grad_op offload`. You
                        can add auto-wrap to `full_shard` or `shard_grad_op` with the same syntax:
                        full_shard auto_wrap` or `shard_grad_op auto_wrap`. (default: )
  --fsdp_min_num_params FSDP_MIN_NUM_PARAMS
                        FSDP's minimum number of parameters for Default Auto Wrapping. (useful only
                        when `fsdp` field is passed). (default: 0)
  --fsdp_transformer_layer_cls_to_wrap FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP
                        Transformer layer class name (case-sensitive) to wrap ,e.g, `BertLayer`,
                        `GPTJBlock`, `T5Block` .... (useful only when `fsdp` flag is passed). (default:
                        None)
  --deepspeed DEEPSPEED
                        Enable deepspeed and pass the path to deepspeed json config file (e.g.
                        ds_config.json) or an already loaded json file as a dict (default: None)
  --label_smoothing_factor LABEL_SMOOTHING_FACTOR
                        The label smoothing epsilon to apply (zero means no label smoothing). (default:
                        0.0)
  --optim {adamw_hf,adamw_torch,adamw_torch_xla,adamw_apex_fused,adafactor,adamw_bnb_8bit,sgd,adagrad}
                        The optimizer to use. (default: adamw_hf)
  --adafactor [ADAFACTOR]
                        Whether or not to replace AdamW by Adafactor. (default: False)
  --group_by_length [GROUP_BY_LENGTH]
                        Whether or not to group samples of roughly the same length together when
                        batching. (default: False)
  --length_column_name LENGTH_COLUMN_NAME
                        Column name with precomputed lengths to use when grouping by length. (default:
                        length)
  --report_to REPORT_TO [REPORT_TO ...]
                        The list of integrations to report the results and logs to. (default: None)
  --ddp_find_unused_parameters DDP_FIND_UNUSED_PARAMETERS
                        When using distributed training, the value of the flag `find_unused_parameters`
                        passed to `DistributedDataParallel`. (default: None)
  --ddp_bucket_cap_mb DDP_BUCKET_CAP_MB
                        When using distributed training, the value of the flag `bucket_cap_mb` passed
                        to `DistributedDataParallel`. (default: None)
  --dataloader_pin_memory [DATALOADER_PIN_MEMORY]
                        Whether or not to pin memory for DataLoader. (default: True)
  --no_dataloader_pin_memory
                        Whether or not to pin memory for DataLoader. (default: False)
  --skip_memory_metrics [SKIP_MEMORY_METRICS]
                        Whether or not to skip adding of memory profiler reports to metrics. (default:
                        True)
  --no_skip_memory_metrics
                        Whether or not to skip adding of memory profiler reports to metrics. (default:
                        False)
  --use_legacy_prediction_loop [USE_LEGACY_PREDICTION_LOOP]
                        Whether or not to use the legacy prediction_loop in the Trainer. (default:
                        False)
  --push_to_hub [PUSH_TO_HUB]
                        Whether or not to upload the trained model to the model hub after training.
                        (default: False)
  --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                        The path to a folder with a valid checkpoint for your model. (default: None)
  --hub_model_id HUB_MODEL_ID
                        The name of the repository to keep in sync with the local `output_dir`.
                        (default: None)
  --hub_strategy {end,every_save,checkpoint,all_checkpoints}
                        The hub strategy to use when `--push_to_hub` is activated. (default:
                        every_save)
  --hub_token HUB_TOKEN
                        The token to use to push to the Model Hub. (default: None)
  --hub_private_repo [HUB_PRIVATE_REPO]
                        Whether the model repository is private or not. (default: False)
  --gradient_checkpointing [GRADIENT_CHECKPOINTING]
                        If True, use gradient checkpointing to save memory at the expense of slower
                        backward pass. (default: False)
  --include_inputs_for_metrics [INCLUDE_INPUTS_FOR_METRICS]
                        Whether or not the inputs will be passed to the `compute_metrics` function.
                        (default: False)
  --fp16_backend {auto,cuda_amp,apex,cpu_amp}
                        Deprecated. Use half_precision_backend instead (default: auto)
  --push_to_hub_model_id PUSH_TO_HUB_MODEL_ID
                        The name of the repository to which push the `Trainer`. (default: None)
  --push_to_hub_organization PUSH_TO_HUB_ORGANIZATION
                        The name of the organization in with to which push the `Trainer`. (default:
                        None)
  --push_to_hub_token PUSH_TO_HUB_TOKEN
                        The token to use to push to the Model Hub. (default: None)
  --mp_parameters MP_PARAMETERS
                        Used by the SageMaker launcher to send mp-specific args. Ignored in Trainer
                        (default: )
  --auto_find_batch_size [AUTO_FIND_BATCH_SIZE]
                        Whether to automatically decrease the batch size in half and rerun the training
                        loop again each time a CUDA Out-of-Memory was reached (default: False)
  --full_determinism [FULL_DETERMINISM]
                        Whether to call enable_full_determinism instead of set_seed for reproducibility
                        in distributed training (default: False)
  --torchdynamo {eager,nvfuser,fx2trt,fx2trt-fp16}
                        Sets up the backend compiler for TorchDynamo. TorchDynamo is a Python level JIT
                        compiler designed to make unmodified PyTorch programs faster. TorchDynamo
                        dynamically modifies the Python bytecode right before its executed. It rewrites
                        Python bytecode to extract sequences of PyTorch operations and lifts them up
                        into Fx graph. We can then pass these Fx graphs to other backend compilers.
                        There are two options - eager and nvfuser. Eager defaults to pytorch eager and
                        is useful for debugging. nvfuser path uses AOT Autograd and nvfuser compiler to
                        optimize the models. (default: None)
  --ray_scope RAY_SCOPE
                        The scope to use when doing hyperparameter search with Ray. By default,
                        `"last"` will be used. Ray will then use the last checkpoint of all trials,
                        compare those, and select the best one. However, other options are also
                        available. See the Ray documentation (https://docs.ray.io/en/latest/tune/api_do
                        cs/analysis.html#ray.tune.ExperimentAnalysis.get_best_trial) for more options.
                        (default: last)
  --ddp_timeout DDP_TIMEOUT
                        Overrides the default timeout for distributed training (value should be given
                        in seconds). (default: 1800)
  --sortish_sampler [SORTISH_SAMPLER]
                        Whether to use SortishSampler or not. (default: False)
  --predict_with_generate [PREDICT_WITH_GENERATE]
                        Whether to use generate to calculate generative metrics (ROUGE, BLEU).
                        (default: False)
  --generation_max_length GENERATION_MAX_LENGTH
                        The `max_length` to use on each evaluation loop when
                        `predict_with_generate=True`. Will default to the `max_length` value of the
                        model configuration. (default: None)
  --generation_num_beams GENERATION_NUM_BEAMS
                        The `num_beams` to use on each evaluation loop when
                        `predict_with_generate=True`. Will default to the `num_beams` value of the
                        model configuration. (default: None)
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
import json
from tqdm import tqdm

import datasets
from evaluate import load as load_metric
import numpy as np
from datasets import load_from_disk

import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    M2M100Tokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.utils.versions import require_version

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

logger = logging.getLogger(__name__)

# A list of all multilingual tokenizer which require src_lang and tgt_lang attributes.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token`."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
                "execute code present on the Hub on your local machine."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_path: str = field(metadata={"help": "Path to the dataset to load from disk (via the datasets library)."})
    metric_name_or_path: str = field(default="sacrebleu")
    source_lang: str = field(default=None, metadata={"help": "Source language id for translation."})
    target_lang: str = field(default=None, metadata={"help": "Target language id for translation."})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the :obj:`decoder_start_token_id`.Useful for"
                " multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token needs to"
                " be the target language token.(Usually it is the target language token)"
            )
        },
    )

    def __post_init__(self):
        if self.source_lang is None or self.target_lang is None:
            raise ValueError("Need to specify the source language and the target language.")

        # accepting both json and jsonl file extensions, as
        # many jsonlines files actually have a .json extension
        valid_extensions = ["json", "jsonl"]

        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in valid_extensions, "`train_file` should be a jsonlines file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in valid_extensions, "`validation_file` should be a jsonlines file."


def args_to_dict(model_args, data_args, training_args):
    model_args = asdict(model_args)
    data_args = asdict(data_args)
    training_args = training_args.to_dict()
    for args in [model_args, data_args]:
        training_args.update(args)
    return training_args


def evaluate(trainer, max_length, num_beams, data_args, eval_dataset):    
    metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
    max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    metrics["num_beams"] = num_beams
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    return metrics


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(exist_ok=True)
    args_dict = args_to_dict(model_args, data_args, training_args)
    with open(output_dir/"args.json", "wt") as file:
        json.dump(args_dict, file, indent=4)
    if model_args.use_auth_token is not None:
        warnings.warn("The `use_auth_token` argument is deprecated and will be removed in v4.34.", FutureWarning)
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = load_from_disk(data_args.dataset_path)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.target_lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # For translation we set the codes of our source and target languages (only useful for mBART, the others will
    # ignore those attributes).
    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert data_args.target_lang is not None and data_args.source_lang is not None, (
            f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --source_lang and "
            "--target_lang arguments."
        )

        tokenizer.src_lang = data_args.source_lang
        tokenizer.tgt_lang = data_args.target_lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

    # Get the language codes for input/target.
    source_lang = data_args.source_lang.split("_")[0]
    target_lang = data_args.target_lang.split("_")[0]

    # Temporarily set max_target_length for training.
    max_length = training_args.generation_max_length
    max_target_length = training_args.generation_max_length
    
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        inputs = [ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Metric
    metric = load_metric(data_args.metric_name_or_path)

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    # load checkpoint only if did not train before
    elif training_args.resume_from_checkpoint is not None:
        trainer._load_from_checkpoint(training_args.resume_from_checkpoint)
    else:
        logger.warn("Did not train nor load checkpoint -> zero-shot evaluation")
    # Evaluation
    results = {}
    num_beams = training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        
        if num_beams == "tune":
            metrics_wrt_beam = []
            for num_beams in tqdm(list(range(1, 10)), desc="Tuning num_beams"):
                metrics = evaluate(trainer, max_length, num_beams, data_args, eval_dataset)
                metrics_wrt_beam.append(metrics)
            with open(output_dir/"metrics_wrt_beam.json", "wt") as file:
                json.dump(metrics_wrt_beam, file)
        else:
            evaluate(trainer, max_length, num_beams, data_args, eval_dataset)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = predict_results.predictions
                predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                predictions = tokenizer.batch_decode(
                    predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    writer.write("\n".join(predictions))

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
