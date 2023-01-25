# https://towardsdatascience.com/how-to-create-and-train-a-multi-task-transformer-model-18c54a146240
from typing import List
from dataclasses import dataclass
from torch import nn
from transformers import AutoTokenizer
from transformers import AutoModel
import torch
from torchvision.ops.focal_loss import sigmoid_focal_loss
from torchinfo import summary
import math

@dataclass
class TaskLayer:
    out_features: int = 0
    in_features: int = 0
    layer_type: str = "Linear"
    activation: str = "ReLU"
    dropout_p: float = None
    # For convolutions
    in_channels: int  = 1       # Number of channels in the input image
    out_channels: int = 1       # Number of channels produced by the convolution
    kernel_size: int  = 3       # Size of the convolving kernel
    stride: int       = 1       # Stride of the convolution. Default: 1
    padding: int      = 0       # Padding added to both sides of the input. Default: 0
    padding_mode: str = 'zeros' # 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    dilation: int     = 1       # Spacing between kernel elements. Default: 1
    groups: int       = 1       # Number of blocked connections from input channels to output channels. Default: 1
    bias: bool        = True    # If True, adds a learnable bias to the output. Default: True


@dataclass
class Task:
    id: int           # A unique task id.
    name: str         # The task name. For printing log messages.
    type: str = "seq_classification" # The task type (seq_classification or token_classification).
    num_labels: int = 2  # The number of labels (i.g., 2 for binary classification).
    problem_type: str = "single_label_classification" # regression, single_label_classification, multi_label_classification
    loss: str = None
    loss_reduction: str = "mean" # mean, sum, none...
    loss_reduction_weight: float = None
    loss_pos_weight: [float] = None
    loss_class_weight: [float] = None
    labels: str = "labels"
    task_layers: [TaskLayer] = None

class MultiTaskModel(nn.Module):
    def __init__(self, encoder_name_or_path, tasks: List):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name_or_path)


        self.output_heads      = nn.ModuleDict()
        #self.output_heads_list = nn.ModuleList()

        self.tasks = tasks
        self.labels_name = {}
        for task in tasks:
            if task.type == "seq_classification_siamese":
                self.encoder2 = AutoModel.from_pretrained(encoder_name_or_path)

            decoder = self._create_output_head(self.encoder.config.hidden_size, task)
            # ModuleDict requires keys to be strings
            self.output_heads[str(task.id)] = decoder
            self.labels_name[str(task.id)] = task.labels
            # self.output_heads_list.append(decoder)

    def freeze(self, requires_grad = True):
        for param in self.encoder.base_model.parameters():
            param.requires_grad = requires_grad
        # for name, param in self.encoder.named_parameters():
        #     if name.startswith(("bert.embeddings", "bert.encoder")):
        #         param.requires_grad = requires_grad
        #     #print(name, param.requires_grad)

    @staticmethod
    def _create_output_head(encoder_hidden_size: int, task):
        if task.type == "seq_classification":
            return SequenceClassificationHead(encoder_hidden_size, task.num_labels, task=task)
        elif task.type == "seq_classification_siamese":
            return TokenClassificationHead(encoder_hidden_size, task.num_labels, task=task)
        elif task.type == "token_classification":
            return TokenClassificationHead(encoder_hidden_size, task.num_labels, task=task)
        else:
            raise NotImplementedError()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_stance=None,
        task_ids=None,
        **kwargs,
    ):
        # print("input_ids:", input_ids)
        # print(task_ids)
        # print(labels_stance, labels_stance.shape)

        if token_type_ids is not None:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

        sequence_output, pooled_output = outputs[:2]

        if task_ids is not None:
            unique_task_ids_list = torch.unique(task_ids).tolist()
        else:
            unique_task_ids_list = [int(i) for i in self.output_heads.keys()]

        loss_list = []
        logits = None
        logits_list = []
        for unique_task_id in unique_task_ids_list:
            task_labels = labels
            match self.labels_name[str(unique_task_id)]:
                case "labels_stance":
                    task_labels = labels_stance
                case _:
                    task_labels = labels

            # task_id_filter = task_ids == unique_task_id
            task_logits, task_loss = self.output_heads[str(unique_task_id)].forward(
                # sequence_output[task_id_filter],
                # pooled_output[task_id_filter],
                sequence_output,
                pooled_output,
                # labels=None if labels is None else labels[task_id_filter],
                labels=None if task_labels is None else task_labels,
                # attention_mask=attention_mask[task_id_filter],
                attention_mask=attention_mask
            )

            if task_logits is not None:
                logits_list.append(task_logits)
            if task_labels is not None:
                loss_list.append(task_loss)
            # if logits is None:
            #     logits = task_logits

        logits = (*logits_list,)
        #print("#logits_list:", logits_list)
        #print("#logits:", logits)
        # logits are only used for eval. and in case of eval the batch is not multi task
        # For training only the loss is used
        outputs = (logits,) + outputs[2:]

        if loss_list:
            # print("loss list:", loss_list)
            loss = torch.stack(loss_list)
            outputs = (loss.mean(),) + outputs
            # outputs = (loss.sum(),) + outputs

        return outputs

class SequenceClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, task=None, dropout_p=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.task = task
        self.layers = torch.nn.ModuleList()

        input_size = hidden_size
        if task.task_layers is not None:
            for tl in task.task_layers:
                ## Dropout...
                if tl.dropout_p is not None:
                    self.layers.append(nn.Dropout(tl.dropout_p))
                match tl.layer_type:
                    case "Conv1d":
                        layer = nn.Conv1d(in_channels=tl.in_channels,
                                                     out_channels=tl.out_channels,
                                                     kernel_size=tl.kernel_size,
                                                     stride=tl.stride,
                                                     padding=tl.padding,
                                                     padding_mode=tl.padding_mode,
                                                     dilation=tl.dilation,
                                                     groups=tl.groups,
                                                     bias=tl.bias)
                        self._init_weights(layer)
                        tl.out_features = math.floor((input_size + 2 * tl.padding - tl.dilation * (tl.kernel_size - 1) - 1) / tl.stride + 1)
                        self.layers.append(layer)
                    case "AvgPool1d":
                        layer = nn.AvgPool1d(tl.kernel_size)
                        self._init_weights(layer)
                        tl.out_features = input_size / tl.kernel_size
                        self.layers.append(layer)
                    case _:
                        if tl.out_features > 0:
                            layer = nn.Linear(input_size, tl.out_features)
                            self._init_weights(layer)
                            self.layers.append(layer)
                input_size = tl.out_features
                if tl.activation is not None:
                    match tl.activation:
                        case "SELU":
                            self.layers.append(nn.SELU())
                        case _:
                            self.layers.append(nn.ReLU())
                input_size = tl.out_features
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(input_size, num_labels)

        self._init_weights()

    def _init_weights(self, layer=None):
        if layer is None:
            layer = self.classifier
        #layer.weight.data.normal_(mean=0.0, std=0.02)
        nn.init.xavier_normal_(layer.weight.data)
        if layer.bias is not None:
            layer.bias.data.zero_()

    def forward(self, sequence_output, pooled_output, labels=None, **kwargs):
        output = pooled_output
        for layer in self.layers:
            if isinstance(layer, nn.Conv1d):
                output = layer(output.unsqueeze(1)).squeeze(1)
            else:
                output = layer(output)
        output = self.dropout(output)
        logits = self.classifier(output)
        # print("=>", self.task.id, self.task.name)
        # print(logits, logits.shape)
        # print(labels, labels.shape)

        loss = None
        if labels is not None:
            # print(labels.dim())
            # if labels.dim() != 1:
            #     # Remove padding
            #     labels = labels[:, 0]

            if self.task is None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.long().view(-1)
                )
            else:
                # print(f"Problem type: {self.task.problem_type}")
                match self.task.problem_type:
                    case "regression":
                        # print(f"Problem type: regression")
                        loss_fct = nn.MSELoss()
                        if self.num_labels == 1:
                            loss = loss_fct(logits.squeeze(), labels.squeeze())
                        else:
                            loss = loss_fct(logits, labels)
                    case "single_label_classification":
                        # print(f"Problem type: single_label_classification")
                        if self.task.loss_class_weight is None:
                            loss_fct = nn.CrossEntropyLoss(reduction=self.task.loss_reduction)
                        else:
                            loss_fct = nn.CrossEntropyLoss(reduction=self.task.loss_reduction, weight=self.task.loss_class_weight.to(logits.device))
                        # Labels are expected as class indices...
                        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                    case "multi_label_classification":
                        # print(f"Problem type: multi_label_classification")
                        # print(logits, logits.shape)
                        # print(labels, labels.shape)
                        reduction = self.task.loss_reduction
                        if self.task.loss_class_weight is not None:
                            ## MultiLabelSoftMarginLoss supports class weights...
                            if not self.task.loss in ["MultiLabelSoftMarginLoss", "SigmoidMultiLabelSoftMarginLoss", "CrossEntropyLoss"]:
                                reduction = "none"

                        match self.task.loss:
                            case "sigmoid_focal_loss":
                                loss_fct = sigmoid_focal_loss
                                loss = loss_fct(logits, labels, reduction=self.task.loss_reduction)
                            case "SigmoidMultiLabelSoftMarginLoss":
                                sigmoid = nn.Sigmoid()
                                logits = sigmoid(logits)
                                loss_fct = nn.MultiLabelSoftMarginLoss(weight=self.task.loss_class_weight.to(logits.device), reduction=reduction)
                                loss = loss_fct(logits, labels)
                            case "MultiLabelSoftMarginLoss":
                                loss_fct = nn.MultiLabelSoftMarginLoss(weight=self.task.loss_class_weight.to(logits.device), reduction=reduction)
                                loss = loss_fct(logits, labels)
                            case "CrossEntropyLoss":
                                ## https://discuss.pytorch.org/t/multilabel-classification-with-class-imbalance/57345
                                # Apply softmax...
                                # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
                                # I think softmax is implicitely applied.
                                #softmax = torch.nn.Softmax(dim=1)
                                #logits = softmax(logits)
                                if self.task.loss_class_weight is None:
                                    loss_fct = nn.CrossEntropyLoss(reduction=self.task.loss_reduction)
                                else:
                                    loss_fct = nn.CrossEntropyLoss(reduction=self.task.loss_reduction, weight=self.task.loss_class_weight.to(logits.device))
                                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
                            case _:
                                loss_fct = nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=self.task.loss_pos_weight.to(logits.device))
                                loss = loss_fct(logits, labels)
                        if reduction == "none":
                            match self.task.loss_reduction:
                                case "sum":
                                    loss = (loss * self.task.loss_class_weight.to(logits.device)).sum()
                                case _:
                                    loss = (loss * self.task.loss_class_weight.to(logits.device)).mean()

                    case _:
                        raise Exception(f"Unknown problem type: {self.task.problem_type} for task {self.task}")

        if loss is not None and self.task.loss_reduction_weight is not None:
            # print(loss, self.task.loss_reduction_weight)
            loss *= self.task.loss_reduction_weight
        # print(loss)
        return logits, loss

class SiameseClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, task=None, dropout_p=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.task = task
        self.layers = torch.nn.ModuleList()

        input_size = hidden_size
        if task.task_layers is not None:
            for tl in task.task_layers:
                ## Dropout...
                if tl.dropout_p is not None:
                    self.layers.append(nn.Dropout(tl.dropout_p))
                if tl.out_features > 0:
                    if tl.linear_features > 0:
                        layer = nn.Linear(input_size, tl.linear_features)
                    else:
                        layer = nn.Linear(input_size, tl.out_features)
                    self._init_weights(layer)
                    self.layers.append(layer)
                match tl.activation:
                    case "SELU":
                        self.layers.append(nn.SELU())
                    case "Conv":
                        self.layers.append(nn.Conv1d(tl.in_features, tl.out_features, 3))
                    case "AvgPool":
                        self.layers.append(nn.AvgPool1d(3))
                    case _:
                        self.layers.append(nn.ReLU())
                input_size = tl.out_features
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(input_size, num_labels)

        self._init_weights()

    def _init_weights(self, layer=None):
        if layer is None:
            layer = self.classifier
        #layer.weight.data.normal_(mean=0.0, std=0.02)
        nn.init.xavier_normal_(layer.weight.data)
        if layer.bias is not None:
            layer.bias.data.zero_()

    def forward(self, sequence_output, pooled_output, labels=None, **kwargs):
        output = pooled_output
        for layer in self.layers:
            if layer is nn.Conv1d:
                output = layer(output.transpose(0, 1))
            else:
                output = layer(output)
        output = self.dropout(output)
        logits = self.classifier(output)
        # print("=>", self.task.id, self.task.name)
        # print(logits, logits.shape)
        # print(labels, labels.shape)

        loss = None
        if labels is not None:
            # print(labels.dim())
            # if labels.dim() != 1:
            #     # Remove padding
            #     labels = labels[:, 0]

            if self.task is None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.long().view(-1)
                )
            else:
                # print(f"Problem type: {self.task.problem_type}")
                match self.task.problem_type:
                    case "regression":
                        # print(f"Problem type: regression")
                        loss_fct = nn.MSELoss()
                        if self.num_labels == 1:
                            loss = loss_fct(logits.squeeze(), labels.squeeze())
                        else:
                            loss = loss_fct(logits, labels)
                    case "single_label_classification":
                        # print(f"Problem type: single_label_classification")
                        if self.task.loss_class_weight is None:
                            loss_fct = nn.CrossEntropyLoss(reduction=self.task.loss_reduction)
                        else:
                            loss_fct = nn.CrossEntropyLoss(reduction=self.task.loss_reduction, weight=self.task.loss_class_weight.to(logits.device))
                        # Labels are expected as class indices...
                        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                    case "multi_label_classification":
                        # print(f"Problem type: multi_label_classification")
                        # print(logits, logits.shape)
                        # print(labels, labels.shape)
                        reduction = self.task.loss_reduction
                        if self.task.loss_class_weight is not None:
                            ## MultiLabelSoftMarginLoss supports class weights...
                            if not self.task.loss in ["MultiLabelSoftMarginLoss", "SigmoidMultiLabelSoftMarginLoss", "CrossEntropyLoss"]:
                                reduction = "none"

                        match self.task.loss:
                            case "sigmoid_focal_loss":
                                loss_fct = sigmoid_focal_loss
                                loss = loss_fct(logits, labels, reduction=self.task.loss_reduction)
                            case "SigmoidMultiLabelSoftMarginLoss":
                                sigmoid = nn.Sigmoid()
                                logits = sigmoid(logits)
                                loss_fct = nn.MultiLabelSoftMarginLoss(weight=self.task.loss_class_weight.to(logits.device), reduction=reduction)
                                loss = loss_fct(logits, labels)
                            case "MultiLabelSoftMarginLoss":
                                loss_fct = nn.MultiLabelSoftMarginLoss(weight=self.task.loss_class_weight.to(logits.device), reduction=reduction)
                                loss = loss_fct(logits, labels)
                            case "CrossEntropyLoss":
                                ## https://discuss.pytorch.org/t/multilabel-classification-with-class-imbalance/57345
                                # Apply softmax...
                                # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
                                # I think softmax is implicitely applied.
                                #softmax = torch.nn.Softmax(dim=1)
                                #logits = softmax(logits)
                                if self.task.loss_class_weight is None:
                                    loss_fct = nn.CrossEntropyLoss(reduction=self.task.loss_reduction)
                                else:
                                    loss_fct = nn.CrossEntropyLoss(reduction=self.task.loss_reduction, weight=self.task.loss_class_weight.to(logits.device))
                                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
                            case _:
                                loss_fct = nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=self.task.loss_pos_weight.to(logits.device))
                                loss = loss_fct(logits, labels)
                        if reduction == "none":
                            match self.task.loss_reduction:
                                case "sum":
                                    loss = (loss * self.task.loss_class_weight.to(logits.device)).sum()
                                case _:
                                    loss = (loss * self.task.loss_class_weight.to(logits.device)).mean()

                    case _:
                        raise Exception(f"Unknown problem type: {self.task.problem_type} for task {self.task}")

        if loss is not None and self.task.loss_reduction_weight is not None:
            # print(loss, self.task.loss_reduction_weight)
            loss *= self.task.loss_reduction_weight
        # print(loss)
        return logits, loss

class TokenClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, task=None, dropout_p=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels
        self.task = task

        self._init_weights()

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def forward(
        self, sequence_output, pooled_output, labels=None, attention_mask=None, **kwargs
    ):
        sequence_output_dropout = self.dropout(sequence_output)
        logits = self.classifier(sequence_output_dropout)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()

            labels = labels.long()

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return logits, loss
