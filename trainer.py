
from typing import NamedTuple, List, Optional, Dict, Tuple, Union, Any
import time
import torch
import numpy as np
import json
import torch.nn as nn

import transformers
from transformers.trainer_utils import PredictionOutput, speed_metrics
from transformers.deepspeed import is_deepspeed_zero3_enabled
from nltk.tokenize import word_tokenize

from packaging import version
if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

from data_utils import YelpTable2TextDataset

class EvalPrediction(NamedTuple):
    predictions: List[str]
    items: List[dict]

class EvaluateFriendlySeq2SeqTrainer(transformers.trainer_seq2seq.Seq2SeqTrainer):
    def __init__(
            self,
            eval_examples: Optional[List] = None,
            compute_metrics = None,
            ignore_pad_token_for_loss: bool = True,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.eval_examples = eval_examples
        self.compute_metrics = compute_metrics 
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss

    def evaluate(
            self,
            eval_dataset: Optional[YelpTable2TextDataset] = None,
            eval_examples: Optional[List] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            max_length: Optional[int] = 450,
            max_time: Optional[int] = None,
            num_beams: Optional[int] = 4,
    ) -> Dict[str, float]:
        self._max_length = max_length
        self._num_beams = num_beams 
        self._max_time = max_time

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples
        start_time = time.time()

        # print([eval_examples[idx]['arg_path'] for idx in range(len(eval_examples))])

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        if eval_examples is not None and eval_dataset is not None and self.compute_metrics is not None:
            eval_preds = self._post_process_function(
                eval_examples,
                output.predictions,
                "eval_{}".format(self.state.epoch) if self.state.epoch is not None else "eval",
            )
            summary = self.compute_metrics(eval_preds, section="dev")
            output.metrics.update(summary)

        n_samples = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(output.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def predict(
            self,
            test_dataset: Optional[YelpTable2TextDataset],
            test_examples: Optional[List],
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            max_length: Optional[int] = 450,
            max_time: Optional[int] = None,
            num_beams: Optional[int] = 4,
    ) -> PredictionOutput:
        self._max_length = max_length 
        self._num_beams = num_beams 
        self._max_time = max_time

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.evaluation_loop(
                test_dataloader,
                description="Prediction",
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.compute_metrics is not None:
            eval_preds = self._post_process_function(
                test_examples, output.predictions, metric_key_prefix)
            output.metrics.update(self.compute_metrics(eval_preds, section="test"))

        output.metrics.update(speed_metrics(metric_key_prefix, start_time, len(test_dataset)))

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(output.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)

        self.log(output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        if prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length,
            "num_beams": self._num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
            "no_repeat_ngram_size": 0,  # FIXME: hard coding the no_repeat_ngram_size
        }

        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            user_labels=inputs["user_labels"] if "user_labels" in inputs else None,
            item_labels=inputs["item_labels"] if "item_labels" in inputs else None,
            **gen_kwargs,
        )

        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, generated_tokens, labels)

    def _post_process_function(
            self, examples: List, predictions: np.ndarray, stage: str
    ) -> EvalPrediction:
        assert isinstance(examples, List)

        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Save locally.
        if self.args.local_rank <= 0:
            out_file = f"{self.args.output_dir}/predictions_{stage}.json"
            with open(out_file, "w") as f:
                json.dump(
                    [dict(**{"prediction": predictions[idx]}, **{"review_text": examples[idx]}) for idx in range(len(predictions))],
                    f,
                    indent=4,
                )
        predictions = [word_tokenize(pred) for pred in predictions]
        # Save to wandb.
        #if self.wandb_run_dir and (
        #        stage.startswith('eval_') and int(self.state.num_train_epochs) == int(float(stage[len('eval_'):]))
        #):
        # if self.args.local_rank <= 0:
        #     with open(f"{self.wandb_run_dir}/predictions_{stage}.json", "w") as f:
        #         json.dump(
        #             [dict(**{"prediction": predictions[idx]}, **examples[idx]) for idx in range(len(predictions))],
        #             f,
        #             indent=4,
        #         )
        return EvalPrediction(
            predictions=predictions, 
            items=[[[word.lower() for word in word_tokenize(examples[idx])[:self._max_length]]] for idx in range(len(predictions))]
        )