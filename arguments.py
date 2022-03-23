"""
The document is used for tracking the training arugments value and its meaning.
"""

import transformers
from dataclasses import dataclass, field
from typing import Optional

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
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    # Added
    num_labels: int = field(
        default=0,
        metadata={
            "help": "The dimension of the output, depends on the number of items to recommend"
        }
    )
    model_type: str = field(
        default='',
        metadata={
            "help": "The base model type (BERT, GPT-2, etc.)"
        }
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    # Added
    yelp_dataset_city: Optional[str] = field(
        default=None, metadata = {"help": "The name of city of yelp dataset that we want to use"}
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file/test_file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."

class TrainingArguments:
    """
    TrainingArguments is the subset of the arguments we use in our example scripts **which relate to the training loop
    itself**.

    Using :class:`~transformers.HfArgumentParser` we can turn this class into `argparse
    <https://docs.python.org/3/library/argparse.html#module-argparse>`__ arguments that can be specified on the command
    line.

    Parameters:
        output_dir (:obj:`str`):
            The output directory where the model predictions and checkpoints will be written.
        overwrite_output_dir (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If :obj:`True`, overwrite the content of the output directory. Use this to continue training if
            :obj:`output_dir` points to a checkpoint directory.
        do_train (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to run training or not. This argument is not directly used by :class:`~transformers.Trainer`, it's
            intended to be used by your training/evaluation scripts instead. See the `example scripts
            <https://github.com/huggingface/transformers/tree/master/examples>`__ for more details.
        do_eval (:obj:`bool`, `optional`):
            Whether to run evaluation on the validation set or not. Will be set to :obj:`True` if
            :obj:`evaluation_strategy` is different from :obj:`"no"`. This argument is not directly used by
            :class:`~transformers.Trainer`, it's intended to be used by your training/evaluation scripts instead. See
            the `example scripts <https://github.com/huggingface/transformers/tree/master/examples>`__ for more
            details.
        do_predict (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to run predictions on the test set or not. This argument is not directly used by
            :class:`~transformers.Trainer`, it's intended to be used by your training/evaluation scripts instead. See
            the `example scripts <https://github.com/huggingface/transformers/tree/master/examples>`__ for more
            details.
        evaluation_strategy (:obj:`str` or :class:`~transformers.trainer_utils.IntervalStrategy`, `optional`, defaults to :obj:`"no"`):
            The evaluation strategy to adopt during training. Possible values are:

                * :obj:`"no"`: No evaluation is done during training.
                * :obj:`"steps"`: Evaluation is done (and logged) every :obj:`eval_steps`.
                * :obj:`"epoch"`: Evaluation is done at the end of each epoch.

        prediction_loss_only (:obj:`bool`, `optional`, defaults to `False`):
            When performing evaluation and generating predictions, only returns the loss.
        per_device_train_batch_size (:obj:`int`, `optional`, defaults to 8):
            The batch size per GPU/TPU core/CPU for training.
        per_device_eval_batch_size (:obj:`int`, `optional`, defaults to 8):
            The batch size per GPU/TPU core/CPU for evaluation.
        gradient_accumulation_steps (:obj:`int`, `optional`, defaults to 1):
            Number of updates steps to accumulate the gradients for, before performing a backward/update pass.

            .. warning::

                When using gradient accumulation, one step is counted as one step with backward pass. Therefore,
                logging, evaluation, save will be conducted every ``gradient_accumulation_steps * xxx_step`` training
                examples.
        eval_accumulation_steps (:obj:`int`, `optional`):
            Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If
            left unset, the whole predictions are accumulated on GPU/TPU before being moved to the CPU (faster but
            requires more memory).
        learning_rate (:obj:`float`, `optional`, defaults to 5e-5):
            The initial learning rate for :class:`~transformers.AdamW` optimizer.
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in
            :class:`~transformers.AdamW` optimizer.
        adam_beta1 (:obj:`float`, `optional`, defaults to 0.9):
            The beta1 hyperparameter for the :class:`~transformers.AdamW` optimizer.
        adam_beta2 (:obj:`float`, `optional`, defaults to 0.999):
            The beta2 hyperparameter for the :class:`~transformers.AdamW` optimizer.
        adam_epsilon (:obj:`float`, `optional`, defaults to 1e-8):
            The epsilon hyperparameter for the :class:`~transformers.AdamW` optimizer.
        max_grad_norm (:obj:`float`, `optional`, defaults to 1.0):
            Maximum gradient norm (for gradient clipping).
        num_train_epochs(:obj:`float`, `optional`, defaults to 3.0):
            Total number of training epochs to perform (if not an integer, will perform the decimal part percents of
            the last epoch before stopping training).
        max_steps (:obj:`int`, `optional`, defaults to -1):
            If set to a positive number, the total number of training steps to perform. Overrides
            :obj:`num_train_epochs`.
        lr_scheduler_type (:obj:`str` or :class:`~transformers.SchedulerType`, `optional`, defaults to :obj:`"linear"`):
            The scheduler type to use. See the documentation of :class:`~transformers.SchedulerType` for all possible
            values.
        warmup_ratio (:obj:`float`, `optional`, defaults to 0.0):
            Ratio of total training steps used for a linear warmup from 0 to :obj:`learning_rate`.
        warmup_steps (:obj:`int`, `optional`, defaults to 0):
            Number of steps used for a linear warmup from 0 to :obj:`learning_rate`. Overrides any effect of
            :obj:`warmup_ratio`.
        log_level (:obj:`str`, `optional`, defaults to ``passive``):
            Logger log level to use on the main process. Possible choices are the log levels as strings: 'debug',
            'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and lets the
            application set the level.
        log_level_replica (:obj:`str`, `optional`, defaults to ``passive``):
            Logger log level to use on replicas. Same choices as ``log_level``"
        log_on_each_node (:obj:`bool`, `optional`, defaults to :obj:`True`):
            In multinode distributed training, whether to log using :obj:`log_level` once per node, or only on the main
            node.
        logging_dir (:obj:`str`, `optional`):
            `TensorBoard <https://www.tensorflow.org/tensorboard>`__ log directory. Will default to
            `output_dir/runs/**CURRENT_DATETIME_HOSTNAME**`.
        logging_strategy (:obj:`str` or :class:`~transformers.trainer_utils.IntervalStrategy`, `optional`, defaults to :obj:`"steps"`):
            The logging strategy to adopt during training. Possible values are:

                * :obj:`"no"`: No logging is done during training.
                * :obj:`"epoch"`: Logging is done at the end of each epoch.
                * :obj:`"steps"`: Logging is done every :obj:`logging_steps`.

        logging_first_step (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to log and evaluate the first :obj:`global_step` or not.
        logging_steps (:obj:`int`, `optional`, defaults to 500):
            Number of update steps between two logs if :obj:`logging_strategy="steps"`.
        save_strategy (:obj:`str` or :class:`~transformers.trainer_utils.IntervalStrategy`, `optional`, defaults to :obj:`"steps"`):
            The checkpoint save strategy to adopt during training. Possible values are:

                * :obj:`"no"`: No save is done during training.
                * :obj:`"epoch"`: Save is done at the end of each epoch.
                * :obj:`"steps"`: Save is done every :obj:`save_steps`.
        save_steps (:obj:`int`, `optional`, defaults to 500):
            Number of updates steps before two checkpoint saves if :obj:`save_strategy="steps"`.
        save_total_limit (:obj:`int`, `optional`):
            If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
            :obj:`output_dir`.
        save_on_each_node (:obj:`bool`, `optional`, defaults to :obj:`False`):
            When doing multi-node distributed training, whether to save models and checkpoints on each node, or only on
            the main one.

            This should not be activated when the different nodes use the same storage as the files will be saved with
            the same names for each node.
        no_cuda (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to not use CUDA even when it is available or not.
        seed (:obj:`int`, `optional`, defaults to 42):
            Random seed that will be set at the beginning of training. To ensure reproducibility across runs, use the
            :func:`~transformers.Trainer.model_init` function to instantiate the model if it has some randomly
            initialized parameters.
        fp16 (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use 16-bit (mixed) precision training instead of 32-bit training.
        fp16_opt_level (:obj:`str`, `optional`, defaults to 'O1'):
            For :obj:`fp16` training, Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details
            on the `Apex documentation <https://nvidia.github.io/apex/amp.html>`__.
        fp16_backend (:obj:`str`, `optional`, defaults to :obj:`"auto"`):
            The backend to use for mixed precision training. Must be one of :obj:`"auto"`, :obj:`"amp"` or
            :obj:`"apex"`. :obj:`"auto"` will use AMP or APEX depending on the PyTorch version detected, while the
            other choices will force the requested backend.
        fp16_full_eval (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use full 16-bit precision evaluation instead of 32-bit. This will be faster and save memory but
            can harm metric values.
        local_rank (:obj:`int`, `optional`, defaults to -1):
            Rank of the process during distributed training.
        tpu_num_cores (:obj:`int`, `optional`):
            When training on TPU, the number of TPU cores (automatically passed by launcher script).
        dataloader_drop_last (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch size)
            or not.
        eval_steps (:obj:`int`, `optional`):
            Number of update steps between two evaluations if :obj:`evaluation_strategy="steps"`. Will default to the
            same value as :obj:`logging_steps` if not set.
        dataloader_num_workers (:obj:`int`, `optional`, defaults to 0):
            Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the
            main process.
        past_index (:obj:`int`, `optional`, defaults to -1):
            Some models like :doc:`TransformerXL <../model_doc/transformerxl>` or :doc:`XLNet <../model_doc/xlnet>` can
            make use of the past hidden states for their predictions. If this argument is set to a positive int, the
            ``Trainer`` will use the corresponding output (usually index 2) as the past state and feed it to the model
            at the next training step under the keyword argument ``mems``.
        run_name (:obj:`str`, `optional`):
            A descriptor for the run. Typically used for `wandb <https://www.wandb.com/>`_ logging.
        disable_tqdm (:obj:`bool`, `optional`):
            Whether or not to disable the tqdm progress bars and table of metrics produced by
            :class:`~transformers.notebook.NotebookTrainingTracker` in Jupyter Notebooks. Will default to :obj:`True`
            if the logging level is set to warn or lower (default), :obj:`False` otherwise.
        remove_unused_columns (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If using :obj:`datasets.Dataset` datasets, whether or not to automatically remove the columns unused by the
            model forward method.

            (Note that this behavior is not implemented for :class:`~transformers.TFTrainer` yet.)
        label_names (:obj:`List[str]`, `optional`):
            The list of keys in your dictionary of inputs that correspond to the labels.

            Will eventually default to :obj:`["labels"]` except if the model used is one of the
            :obj:`XxxForQuestionAnswering` in which case it will default to :obj:`["start_positions",
            "end_positions"]`.
        load_best_model_at_end (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to load the best model found during training at the end of training.

            .. note::

                When set to :obj:`True`, the parameters :obj:`save_strategy` needs to be the same as
                :obj:`eval_strategy`, and in the case it is "steps", :obj:`save_steps` must be a round multiple of
                :obj:`eval_steps`.
        metric_for_best_model (:obj:`str`, `optional`):
            Use in conjunction with :obj:`load_best_model_at_end` to specify the metric to use to compare two different
            models. Must be the name of a metric returned by the evaluation with or without the prefix :obj:`"eval_"`.
            Will default to :obj:`"loss"` if unspecified and :obj:`load_best_model_at_end=True` (to use the evaluation
            loss).

            If you set this value, :obj:`greater_is_better` will default to :obj:`True`. Don't forget to set it to
            :obj:`False` if your metric is better when lower.
        greater_is_better (:obj:`bool`, `optional`):
            Use in conjunction with :obj:`load_best_model_at_end` and :obj:`metric_for_best_model` to specify if better
            models should have a greater metric or not. Will default to:

            - :obj:`True` if :obj:`metric_for_best_model` is set to a value that isn't :obj:`"loss"` or
              :obj:`"eval_loss"`.
            - :obj:`False` if :obj:`metric_for_best_model` is not set, or set to :obj:`"loss"` or :obj:`"eval_loss"`.
        ignore_data_skip (:obj:`bool`, `optional`, defaults to :obj:`False`):
            When resuming training, whether or not to skip the epochs and batches to get the data loading at the same
            stage as in the previous training. If set to :obj:`True`, the training will begin faster (as that skipping
            step can take a long time) but will not yield the same results as the interrupted training would have.
        sharded_ddp (:obj:`bool`, :obj:`str` or list of :class:`~transformers.trainer_utils.ShardedDDPOption`, `optional`, defaults to :obj:`False`):
            Use Sharded DDP training from `FairScale <https://github.com/facebookresearch/fairscale>`__ (in distributed
            training only). This is an experimental feature.

            A list of options along the following:

            - :obj:`"simple"`: to use first instance of sharded DDP released by fairscale (:obj:`ShardedDDP`) similar
              to ZeRO-2.
            - :obj:`"zero_dp_2"`: to use the second instance of sharded DPP released by fairscale
              (:obj:`FullyShardedDDP`) in Zero-2 mode (with :obj:`reshard_after_forward=False`).
            - :obj:`"zero_dp_3"`: to use the second instance of sharded DPP released by fairscale
              (:obj:`FullyShardedDDP`) in Zero-3 mode (with :obj:`reshard_after_forward=True`).
            - :obj:`"offload"`: to add ZeRO-offload (only compatible with :obj:`"zero_dp_2"` and :obj:`"zero_dp_3"`).

            If a string is passed, it will be split on space. If a bool is passed, it will be converted to an empty
            list for :obj:`False` and :obj:`["simple"]` for :obj:`True`.
        deepspeed (:obj:`str` or :obj:`dict`, `optional`):
            Use `Deepspeed <https://github.com/microsoft/deepspeed>`__. This is an experimental feature and its API may
            evolve in the future. The value is either the location of DeepSpeed json config file (e.g.,
            ``ds_config.json``) or an already loaded json file as a :obj:`dict`"
        label_smoothing_factor (:obj:`float`, `optional`, defaults to 0.0):
            The label smoothing factor to use. Zero means no label smoothing, otherwise the underlying onehot-encoded
            labels are changed from 0s and 1s to :obj:`label_smoothing_factor/num_labels` and :obj:`1 -
            label_smoothing_factor + label_smoothing_factor/num_labels` respectively.
        debug (:obj:`str` or list of :class:`~transformers.debug_utils.DebugOption`, `optional`, defaults to :obj:`""`):
            Enable one or more debug features. This is an experimental feature.

            Possible options are:

            - :obj:`"underflow_overflow"`: detects overflow in model's input/outputs and reports the last frames that
              led to the event
            - :obj:`"tpu_metrics_debug"`: print debug metrics on TPU

            The options should be separated by whitespaces.
        adafactor (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use the :class:`~transformers.Adafactor` optimizer instead of
            :class:`~transformers.AdamW`.
        group_by_length (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to group together samples of roughly the same length in the training dataset (to minimize
            padding applied and be more efficient). Only useful if applying dynamic padding.
        length_column_name (:obj:`str`, `optional`, defaults to :obj:`"length"`):
            Column name for precomputed lengths. If the column exists, grouping by length will use these values rather
            than computing them on train startup. Ignored unless :obj:`group_by_length` is :obj:`True` and the dataset
            is an instance of :obj:`Dataset`.
        report_to (:obj:`str` or :obj:`List[str]`, `optional`, defaults to :obj:`"all"`):
            The list of integrations to report the results and logs to. Supported platforms are :obj:`"azure_ml"`,
            :obj:`"comet_ml"`, :obj:`"mlflow"`, :obj:`"tensorboard"` and :obj:`"wandb"`. Use :obj:`"all"` to report to
            all integrations installed, :obj:`"none"` for no integrations.
        ddp_find_unused_parameters (:obj:`bool`, `optional`):
            When using distributed training, the value of the flag :obj:`find_unused_parameters` passed to
            :obj:`DistributedDataParallel`. Will default to :obj:`False` if gradient checkpointing is used, :obj:`True`
            otherwise.
        dataloader_pin_memory (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether you want to pin memory in data loaders or not. Will default to :obj:`True`.
        skip_memory_metrics (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to skip adding of memory profiler reports to metrics. This is skipped by default because it slows
            down the training and evaluation speed.
        push_to_hub (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to upload the trained model to the hub after training. If this is activated, and
            :obj:`output_dir` exists, it needs to be a local clone of the repository to which the
            :class:`~transformers.Trainer` will be pushed.
        resume_from_checkpoint (:obj:`str`, `optional`):
            The path to a folder with a valid checkpoint for your model. This argument is not directly used by
            :class:`~transformers.Trainer`, it's intended to be used by your training/evaluation scripts instead. See
            the `example scripts <https://github.com/huggingface/transformers/tree/master/examples>`__ for more
            details.
        push_to_hub_model_id (:obj:`str`, `optional`):
            The name of the repository to which push the :class:`~transformers.Trainer` when :obj:`push_to_hub=True`.
            Will default to the name of :obj:`output_dir`.
        push_to_hub_organization (:obj:`str`, `optional`):
            The name of the organization in with to which push the :class:`~transformers.Trainer`.
        push_to_hub_token (:obj:`str`, `optional`):
            The token to use to push the model to the Hub. Will default to the token in the cache folder obtained with
            :obj:`huggingface-cli login`.
    """