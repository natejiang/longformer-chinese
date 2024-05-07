import os

import json
import os
import random
import argparse
from argparse import Namespace
import numpy as np
import glob
import gzip

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import torch.distributed as dist

from longformer.longformer import Longformer, LongformerConfig
from longformer.sliding_chunks import pad_to_window_size
from transformers import LongformerModel,BertTokenizer, AdamW

from torch.utils.data.dataset import IterableDataset
from tqdm.auto import tqdm

import logging
logger = logging.getLogger(__name__)

from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from custom_dataset import CustomDataset

TEXT_FIELD_NAME = 'text'
LABEL_FIELD_NAME = 'label'

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    # '': get_constant_schedule,             # not supported for now
    # '': get_constant_schedule_with_warmup, # not supported for now
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"

def calc_f1(y_pred:torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
    """
    计算给定预测值和真实值的F1分数。

    参数:
    y_pred:torch.Tensor - 预测标签的张量，形状与y_true相同。
    y_true:torch.Tensor - 真实标签的张量，形状与y_pred相同。

    返回值:
    torch.Tensor - 计算得到的F1分数，为一个张量。

    """

    # 计算真正例、假正例、真负例和假负例的数量
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7  # 为了避免除以零的错误，设置一个非常小的正数

    # 计算精确度和召回率
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    # 根据精确度和召回率计算F1分数
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1 = f1.clamp(min=epsilon, max=1 - epsilon)  # 确保F1分数在[0, 1]范围内

    return f1


class ClassificationDataset(Dataset):
    def __init__(self, file_path, tokenizer, seqlen, num_samples=None, mask_padding_with_zero=True):
        self.data = []
        with (gzip.open(file_path, 'rt') if file_path.endswith('.gz') else open(file_path)) as fin:
            for i, line in enumerate(tqdm(fin, desc=f'loading input file {file_path.split("/")[-1]}', unit_scale=1)):
                items = line.strip().split('\tSEP\t')
                if len(items) != 10: continue
                self.data.append({
                    "text": items[0]+items[1],
                    "label": items[5]
                })
                if num_samples and len(self.data) > num_samples:
                    break
        self.seqlen = seqlen
        self._tokenizer = tokenizer
        all_labels = list(set([e[LABEL_FIELD_NAME] for e in self.data]))
        self.label_to_idx = {e: i for i, e in enumerate(sorted(all_labels))}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.mask_padding_with_zero = mask_padding_with_zero

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self._convert_to_tensors(self.data[idx])

    def _convert_to_tensors(self, instance):
        def tok(s):
            return self._tokenizer.tokenize(s)
        tokens = [self._tokenizer.cls_token] + tok(instance[TEXT_FIELD_NAME])
        token_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        token_ids = token_ids[:self.seqlen-1] +[self._tokenizer.sep_token_id]
        input_len = len(token_ids)
        attention_mask = [1 if self.mask_padding_with_zero else 0] * input_len
        padding_length = self.seqlen - input_len
        token_ids = token_ids + ([self._tokenizer.pad_token_id] * padding_length)

        attention_mask = attention_mask + ([0 if self.mask_padding_with_zero else 1] * padding_length)

        assert len(token_ids) == self.seqlen, "Error with input length {} vs {}".format(
            len(token_ids), self.seqlen
        )
        assert len(attention_mask) == self.seqlen, "Error with input length {} vs {}".format(
            len(attention_mask), self.seqlen
        )

        label = self.label_to_idx[instance[LABEL_FIELD_NAME]]

        return (torch.tensor(token_ids), torch.tensor(attention_mask), torch.tensor(label))


class LongformerClassifier(pl.LightningModule):

    def __init__(self, init_args):
        """
        初始化函数。
        
        参数:
        - init_args: 初始化参数，可以是一个字典或具有属性的对象，字典应包含模型配置和检查点路径等信息。
        
        该函数负责加载模型配置、检查点、分词器，并根据初始化参数设置模型的注意模式和序列长度，
        以及准备用于验证步骤的输出列表。
        """
        super().__init__()
        if isinstance(init_args, dict):
            # 将字典参数转换为Namespace对象，以便于属性访问
            init_args = Namespace(**init_args)
        # 获取模型配置和检查点路径
        config_path = init_args.config_path or init_args.model_dir
        checkpoint_path = init_args.checkpoint_path or init_args.model_dir
        logger.info(f'loading model from config: {config_path}, checkpoint: {checkpoint_path}')
        # 从预训练配置加载模型配置，并设置注意模式
        config = LongformerConfig.from_pretrained(config_path)
        config.attention_mode = init_args.attention_mode
        logger.info(f'attention mode set to {config.attention_mode}')
        self.model_config = config
        # 从预训练检查点加载模型
        self.model = Longformer.from_pretrained(checkpoint_path, config=config)
        # 加载并配置分词器
        self.tokenizer = BertTokenizer.from_pretrained(init_args.tokenizer)
        self.tokenizer.model_max_length = self.model.config.max_position_embeddings
        # 保存超参数
        self.save_hyperparameters(init_args)
        # 设置序列长度和分类器
        self.hparams.seqlen = self.model.config.max_position_embeddings
        # self.hparams.seqlen = 512
        self.classifier = nn.Linear(config.hidden_size, init_args.num_labels)

        # 初始化用于存储验证步骤输出的列表
        self.validation_step_outputs = []

        # 初始化验证步骤的预测结果和标签列表
        self.validation_step_preds = []  # 用于存储验证步骤中的预测结果
        self.validation_step_labels = []  # 用于存储验证步骤中的真实标签

    def forward(self, input_ids, attention_mask, labels=None):
        """
        前向传播函数，用于处理输入数据并计算模型输出和损失（如果提供了标签）。
        
        参数:
        - input_ids: 输入序列的标识符张量。
        - attention_mask: 表示哪些位置是有效注意力位置的掩码。
        - labels: (可选) 目标标签的张量。如果提供，将计算并返回损失。
        
        返回:
        - logits: 模型预测的未经过softmax激活的概率分布。
        - loss: (可选) 根据提供的标签计算的损失值。如果未提供标签，则不返回。
        """
        # 将输入_ids和attention_mask调整到固定的窗口大小，并填充
        input_ids, attention_mask = pad_to_window_size(
            input_ids, attention_mask, self.model_config.attention_window[0], self.tokenizer.pad_token_id)
        attention_mask[:, 0] = 2  # 为第一个token设置全局注意力
        
        # 打印调整后的input_ids和attention_mask，用于调试
        # print("input_ids:", input_ids)
        # print("attention_mask:", attention_mask)
        
        # 通过模型进行前向传播
        output = self.model(input_ids, attention_mask=attention_mask)[1]
        # 注意: 这里输出被池化为一个向量(CLS token)，但当前注释掉了
        
        # 通过分类器层获取最终的logits
        logits = self.classifier(output)

        # 初始化损失为None
        loss = None
        # 如果提供了标签，则计算损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            loss = loss_fct(logits.view(-1, self.hparams.num_labels), labels.view(-1).type(torch.long))

        return logits, loss

    def _get_loader(self, split, shuffle=True):
        if split == 'train':
            fname = self.hparams.train_file
            dataset = CustomDataset(
            fname, tokenizer=self.tokenizer, seqlen=self.hparams.seqlen, num_samples=self.hparams.num_samples
            )
        elif split == 'dev':
            fname = self.hparams.dev_file
            dataset = CustomDataset(
            fname, tokenizer=self.tokenizer, seqlen=self.hparams.seqlen
        )
        elif split == 'test':
            fname = self.hparams.test_file
            dataset = CustomDataset(
            fname, tokenizer=self.tokenizer, seqlen=self.hparams.seqlen
        )
        else:
            assert False
        is_train = split == 'train'

        loader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=(shuffle and is_train))
        return loader

    def setup(self, stage='fit'):
        """
        初始化函数，为特定的阶段设置训练数据加载器。
        
        参数:
        - self: 对象自身的引用。
        - stage: 字符串，指定设置的阶段，默认为'fit'。当前阶段限定于'fit'，但设计上可能支持更多阶段。
        
        返回值:
        - 无。
        """
        self.train_loader = self._get_loader("train") # 获取训练数据的加载器，为训练（fit）阶段准备数据。

    def train_dataloader(self):
        """
        获取训练数据加载器
        
        该方法用于返回实例化的训练数据加载器。
        
        参数:
        self - 对象本身的引用
        
        返回值:
        train_loader - 训练数据的加载器，用于在训练过程中批量加载数据
        """
        return self.train_loader

    def val_dataloader(self):
        self.val_dataloader_obj = self._get_loader('dev')
        return self.val_dataloader_obj

    def test_dataloader(self):
        return self._get_loader('test')

    @property
    def total_steps(self) -> int:
        """
        计算总共需要执行的训练步数。此方法用于学习率调度器的目的。
        
        参数:
        self: 对象自身的引用。
        
        返回值:
        int: 总训练步数。
        """
        # 根据提供的GPU数量，确定使用的设备数量，至少为1
        num_devices = max(1, self.hparams.total_gpus)  
        # 计算有效批次大小，考虑了梯度累积和设备数量
        effective_batch_size = self.hparams.batch_size * self.hparams.grad_accum * num_devices
        # 获取训练数据集的大小
        dataset_size = len(self.train_loader.dataset)
        # 根据数据集大小、有效批次大小和训练周期数计算总步数
        return (dataset_size / effective_batch_size) * self.hparams.num_epochs

    def get_lr_scheduler(self):
        """
        获取学习率调度器。
        
        根据预设的参数（hparams.lr_scheduler）选择适当的学习率调度函数（arg_to_scheduler映射），
        并配置该调度器，包括预热步数（warmup_steps）和总训练步数（total_steps）。
        
        返回值:
            一个字典，包含学习率调度器对象及其调度间隔和频率的配置。
        """
        # 根据预设的学习率调度器名称，从映射中获取对应的调度函数
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        # 调用调度函数，配置预热步数和总训练步数
        scheduler = get_schedule_func(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        # 将调度器封装成标准格式，指定调度器的调用间隔和频率
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """
        配置优化器和学习率计划（线性预热和衰减）。
        
        函数不接受参数。
        
        返回:
        - 优化器列表: 包含单个优化器的列表。
        - 调度器列表: 包含单个调度器的列表，用于控制学习率的变化。
        """
        model = self.model
        # 不进行权重衰减的参数名称列表
        no_decay = ["bias", "LayerNorm.weight"]
        # 分组参数，分为两组：一组允许权重衰减，一组不允许
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
                "weight_decay": self.hparams.weight_decay,  # 应用于允许权重衰减的参数组
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
                "weight_decay": 0.0,  # 不应用于不允许权重衰减的参数组
            },
        ]
        # 根据是否使用 Adafactor 选择优化器
        if self.hparams.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters, lr=self.hparams.lr, scale_parameter=False, relative_step=False
            )
        else:
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=self.hparams.lr, eps=self.hparams.adam_epsilon
            )
        self.opt = optimizer  # 存储优化器实例

        scheduler = self.get_lr_scheduler()  # 获取学习率调度器

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        """
        执行训练步骤，包括前向传播、损失计算和日志记录。
        
        参数:
        - batch: 一个包含了输入数据、注意力掩码和标签的批次。
        - batch_idx: 批次索引，用于记录当前批次的编号。
        
        返回值:
        - 一个字典，包含了损失值和要记录到TensorBoard的日志信息。
        """
        
        # 准备输入数据
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

        # 前向传播，计算输出和损失
        outputs = self(**inputs)
        loss = outputs[1]

        # 获取优化器和学习率调度器
        scheduler = self.lr_schedulers() 

        # 计算并记录当前学习率
        tensorboard_logs = {"loss": loss, "rate": scheduler.get_last_lr()[-1]}

        return {"loss": loss, "log": tensorboard_logs}


    def validation_step(self, batch, batch_idx):
        """
        执行单个验证步骤。
        
        参数:
        - batch: 一个包含了输入数据、注意力掩码和标签的批次。其中，输入数据和注意力掩码用于模型推断，标签用于计算损失和评估。
        - batch_idx: 批次索引，表示当前处理的是第几个批次。
        
        返回:
        - 一个字典，包含了验证阶段的损失（val_loss）、预测结果（pred）和真实标签（target）。
        """
        
        # 准备输入数据，包括输入_ids、注意力掩码和标签
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

        # 使用模型进行预测，获取模型输出的logits和评估损失
        outputs = self(**inputs)
        logits, tmp_eval_loss = outputs
        preds = logits
        out_label_ids = inputs["labels"]
        
        # 将当前批次的评估损失添加到存储列表中
        self.validation_step_outputs.append(tmp_eval_loss)
        
        self.validation_step_preds.append(preds)  # 收集预测结果
        self.validation_step_labels.append(out_label_ids)  # 收集对应的标签
    
        # 返回损失、预测结果和真实标签
        return {"val_loss": tmp_eval_loss, "pred": preds, "target": out_label_ids}


    def _eval_end(self, loss_list, preds_list, labels_list) -> tuple:
        """
        计算并返回评估结束时的指标结果。

        此函数计算给定输出列表的平均损失、准确度和F1分数，并针对使用分布式数据并行的情况进行指标的聚合。
        
        参数:
        - outputs: 一个包含多个步骤输出的列表，每个输出是一个字典，包含‘val_loss’、‘pred’、‘target’键。
        
        返回值:
        - 一个字典，包含‘val_loss’（平均损失）、'f1'（F1分数）、'acc'（准确度）和‘log’（包含所有结果的字典）键值对。
        """
        # 计算平均损失
        avg_loss = torch.stack(loss_list).mean()

        # 合并预测和标签
        preds = torch.cat(preds_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        # 获取预测类别
        preds = torch.argmax(preds, axis=-1)
        print('------------------------------------')
        print('preds: ',preds)
        print('------------------------------------')
        print('labels: ',labels)
        print('------------------------------------')
        # 计算准确度
        accuracy = (preds == labels).int().sum() / float(torch.tensor(preds.shape[-1], dtype=torch.float32, device=labels.device))
        # 计算F1分数
        f1 = calc_f1(preds, labels)
        # 如果使用分布式数据并行，对损失和指标进行聚合
        if self.trainer.accelerator in ('ddp', 'ddp_cpu', 'ddp_spawn', 'gpu') and self.trainer.is_global_zero:
            torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.SUM)
            avg_loss /= self.trainer.world_size
            torch.distributed.all_reduce(accuracy, op=torch.distributed.ReduceOp.SUM)
            accuracy /= self.trainer.world_size
            torch.distributed.all_reduce(f1, op=torch.distributed.ReduceOp.SUM)
            f1 /= self.trainer.world_size
        # 构建结果字典
        results = {"val_loss": avg_loss, "f1": f1, "acc": accuracy}
        self.log('acc', accuracy)
        # 包含所有结果的字典，用于日志记录等
        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret

    def on_validation_epoch_end(self) -> dict:
        ret = self._eval_end(self.validation_step_outputs, self.validation_step_preds, self.validation_step_labels)
        logs = ret["log"]
        return {"val_loss": logs["val_loss"], "log": logs, "progress_bar": logs}


    def on_test_epoch_end(self) -> dict:
        """
        在测试阶段结束时调用的方法，用于处理和返回测试结果的汇总信息。
        
        参数:
        - outputs: 一个列表，包含了模型在测试批次上的输出结果。
        
        返回值:
        - 一个字典，包含了平均测试损失、日志信息和进度条信息。
        """
        # 调用_eval_end方法处理outputs，并获取日志和其他结果
        ret = self._eval_end(self.validation_step_outputs, self.validation_step_preds, self.validation_step_labels)
        logs = ret["log"]
        results = {}
        
        # 遍历日志项，将torch.Tensor类型的值转换为cpu上的数值
        for k, v in logs.items():
            if isinstance(v, torch.Tensor):
                results[k] = v.detach().cpu().item()
        
        # 返回测试损失、日志结果和进度条信息
        # 注意: val_loss 实际上指的是测试损失(test_loss)
        return {"avg_test_loss": logs["val_loss"].detach().cpu().item(), "log": results, "progress_bar": results}

    def test_step(self, batch, batch_nb):
        """
        执行测试步骤，实际上是调用了验证步骤。

        参数:
        self: 对象自身的引用。
        batch: 一个批次的数据，用于测试模型。
        batch_nb: 批次的编号，用于记录或日志。

        返回:
        调用validation_step函数的返回值，通常包含模型在该批次数据上的验证结果。
        """
        return self.validation_step(batch, batch_nb)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', dest='model_dir', default='D:\\models\\longformer-chinese-base-4096', help='path to the model')
    parser.add_argument('--config_path', default=None, help='path to the config (if not setting dir)')
    parser.add_argument('--checkpoint_path', default=None, help='path to the model (if not setting checkpoint)')
    parser.add_argument('--attention_mode', required=True, default='sliding_chunks')
    parser.add_argument('--tokenizer', default='D:\\models\\longformer-chinese-base-4096')
    parser.add_argument('--train_file')
    parser.add_argument('--dev_file')
    parser.add_argument('--test_file')
    parser.add_argument('--input_dir', default=None, help='optionally provide a directory of the data and train/test/dev files will be automatically detected')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--grad_accum', default=1, type=int)
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--seed', default=2012, type=int)
    parser.add_argument('--fp16', default=False, action='store_true')
    parser.add_argument('--test_only', default=False, action='store_true')
    parser.add_argument('--test_checkpoint', default=None)
    parser.add_argument('--test_percent_check', default=1.0, type=float)
    parser.add_argument('--limit_val_batches', default=1.0, type=float)
    parser.add_argument('--val_check_interval', default=1.0, type=float)
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--do_predict', default=True, action='store_true')
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--num_workers", default=4, type=int, help="kwarg passed to DataLoader")
    parser.add_argument("--adafactor", action="store_true")
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--num_labels', default=-1, type=int,
        help='if -1, it automatically finds number of labels.'
        'for larger datasets precomute this and manually set')
    parser.add_argument('--num_samples', default=None, type=int)
    parser.add_argument("--lr_scheduler",
        default="linear",
        choices=arg_to_scheduler_choices,
        metavar=arg_to_scheduler_metavar,
        type=str,
        help="Learning rate scheduler")
    args = parser.parse_args()

    if args.input_dir is not None:
        files = glob.glob(args.input_dir + '/*')
        for f in files:
            fname = f.split('/')[-1]
            if 'train' in fname:
                args.train_file = f
            elif 'dev' in fname or 'val' in fname:
                args.dev_file = f
            elif 'test' in fname:
                args.test_file = f
    return args

def get_train_params(args):
    """
    根据传入的参数构建训练参数字典。

    参数:
    - args: 一个包含各种训练参数的对象。预期包含fp16、gpus、grad_accum、limit_val_batches、val_check_interval和num_epochs属性。

    返回值:
    - 一个字典，包含训练过程中的关键参数设置，如精度、加速器类型、梯度累积步数等。
    """
    train_params = {}
    # 设置精度，根据是否使用fp16来决定是16位还是32位
    train_params["precision"] = 16 if args.fp16 else 32
    # 根据gpu数量决定使用分布式数据并行(ddp)还是cuda加速
    if (isinstance(args.gpus, int) and args.gpus > 1) or (isinstance(args.gpus, list ) and len(args.gpus) > 1):
        train_params["accelerator"] = "gpu"
    else:
        train_params["accelerator"] = "cuda"
    # 设置梯度累积步数
    train_params["accumulate_grad_batches"] = args.grad_accum
    # 设置验证集的限制批次和检查间隔
    train_params['limit_val_batches'] = args.limit_val_batches
    train_params['val_check_interval'] = args.val_check_interval
    # 设定使用的设备
    train_params['devices'] = args.gpus
    # 设定最大训练周期数
    train_params['max_epochs'] = args.num_epochs
    return train_params

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if ',' in args.gpus:
        args.gpus = list(map(int, args.gpus.split(',')))
        args.total_gpus = len(args.gpus)
    else:
        args.gpus = int(args.gpus)
        args.total_gpus = args.gpus

    def infer_num_labels(args):
        """
        根据训练文件和分词器估计标签的数量。
        
        参数:
        - args: 一个包含训练文件路径和分词器信息的对象。其中`train_file`属性指定了训练文件的路径，`tokenizer`属性指定了用于分词的分词器。
        
        返回值:
        - num_labels: 整数，表示训练数据集中不同标签的数量。
        """
        # 使用自定义数据集类加载训练数据，仅读取标签，不关心序列长度
        ds = CustomDataset(args.train_file, tokenizer=args.tokenizer, seqlen=4096)
        # 计算并返回标签的数量
        num_labels = len(ds.label_to_idx)
        return num_labels

    # 当参数 args.test_only 为 True 时，执行测试模式
    if args.test_only:
        print('loading model...')  # 打印加载模型信息
        # 如果标签数量未指定，则推断标签数量
        if args.num_labels == -1:
            args.num_labels = infer_num_labels(args)
        # 从指定检查点加载 Longformer 分类器模型
        model = LongformerClassifier.load_from_checkpoint(args.test_checkpoint, num_labels=args.num_labels)
        # 设置训练器参数，包括使用的 GPU 数量和测试数据的百分比
        trainer = pl.Trainer(gpus=args.gpus, test_percent_check=args.test_percent_check)
        # 执行模型测试
        trainer.test(model)

    else:
        if args.num_labels == -1:
            args.num_labels = infer_num_labels(args)
        model = LongformerClassifier(args)

        # default logger used by trainer
        logger = TensorBoardLogger(
            save_dir=args.save_dir,
            version=0,
            name='pl-logs'
        )

        # second part of the path shouldn't be f-string
        # filepath = f'{args.save_dir}/version_{logger.version}/checkpoints/' + 'ep-{epoch}_acc-{acc:.3f}'
        dirpath = f'{args.save_dir}/version_{logger.version}/checkpoints/'
        filename = 'ep-{epoch}_acc-{acc:.3f}'

        """
        创建一个模型检查点回调对象，用于在训练过程中定期保存最佳模型。

        参数:
        - dirpath: 指定保存模型的目录路径。
        - filename: 指定模型文件的基础名称。
        - save_top_k: 保存最佳模型的数量，此处设置为10。
        - verbose: 是否输出保存模型时的详细信息，设置为True。
        - monitor: 监控的指标，用于决定保存哪个模型，此处设置为'acc'（准确率）。
        - mode: 监控指标的优化模式，'max'表示最大化监控指标，'min'表示最小化监控指标。
        """
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            filename=filename,
            save_top_k=10,
            verbose=True,
            monitor='acc',
            mode='max',
            # prefix=''
        )

        extra_train_params = get_train_params(args)

        trainer = pl.Trainer(logger=logger,
                            callbacks=[checkpoint_callback],
                            gradient_clip_val=1.0,  # 设置梯度裁剪的阈值
                            gradient_clip_algorithm="norm",  # 选择裁剪算法，这里是按范数裁剪
                            **extra_train_params)

        trainer.fit(model)

        if args.do_predict:
            """
            如果设置了预测标志，则执行预测步骤。
            此部分代码首先从指定路径加载最新的模型检查点，然后配置模型以使用单个GPU进行预测。
            随后，利用PyTorch Lightning的Trainer进行模型的测试。
            """
            # 从指定目录获取第一个.ckpt文件路径
            fpath = glob.glob(checkpoint_callback.dirpath + '/*.ckpt')[0]
            args.checkpoint_path = fpath
            # 从检查点加载Longformer分类器模型
            model = LongformerClassifier.load_from_checkpoint(fpath)
            # 配置模型以使用单个GPU
            model.hparams.num_gpus = 1
            model.hparams.total_gpus = 1
            # 更新模型的超参数为命令行传入的参数
            # model.hparams = args
            # 设置开发集、测试集文件路径
            model.hparams.dev_file = args.dev_file
            model.hparams.test_file = args.test_file
            # 为了快速加载，将开发集文件路径设置为训练文件路径
            model.hparams.train_file = args.dev_file
            # 配置Trainer，使用1个GPU，设置测试数据的检查比例和训练数据的检查比例
            trainer = pl.Trainer(devices=1, 
                                 limit_val_batches=0.1, precision=extra_train_params['precision'])
            # 在测试模式下运行模型
            trainer.test(model)

if __name__ == '__main__':
    main()
