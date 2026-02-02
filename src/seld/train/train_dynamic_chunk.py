import logging
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.config_manager import C, init_config, parse_cli_args
from seld.lr_scheduler.tri_stage_lr_scheduler import TriStageLRScheduler
from seld.models.cache_resnet_conformer import ResnetConformer_sed_doa_nopool
from seld.utils.dataset.lmdb_data_loader_A import LmdbDataset
from seld.utils.feature.compute_seld_results import ComputeSELDResults
from seld.utils.process import SetRandomSeed, write_output_format_file
from seld.utils.result.sed_doa import (
    SedDoaLoss,
    SedDoaResult,
    process_foa_input_sed_doa_labels,
)


def main():
    # 1. 解析命令行参数 (获取配置文件路径和覆盖项)
    #    例如运行: python train_dynamic_chunk.py --config configs/exp1.toml train.lr=0.001
    cli_args = parse_cli_args()

    # 2. 初始化全局配置
    #    这会加载配置文件，并应用命令行里的覆盖项
    init_config(**cli_args)

    log_dir = os.path.dirname(C().result.log_output_path)
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=C().result.log_output_path,
        filemode="w",
        level=logging.INFO,
        format="%(levelname)s: %(asctime)s: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.info(C())

    criterion = SedDoaLoss(loss_weight=[0.1, 1])
    model = ResnetConformer_sed_doa_nopool(
        in_channel=C().model.in_channel,
        in_dim=C().model.in_dim,
        out_dim=C().model.out_dim,
        att_context_size=C().model.att_context_size,
        num_conformer_layer=C().model.num_conformer_layers,
        encoder_dim=C().model.encoder_dim,
    )

    train_split = [1, 2, 3]
    train_dataset = LmdbDataset(
        C().data.train_lmdb_dir,
        train_split,
        normalized_features_wts_file=C().data.norm_file,
        ignore=C().data.train_ignore,
        segment_len=C().data.segment_len,
        data_process_fn=process_foa_input_sed_doa_labels,
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=C().data.batch_size,
        shuffle=True,
        num_workers=C().train.train_num_workers,
        collate_fn=train_dataset.collater,
    )

    test_split = [4]
    test_dataset = LmdbDataset(
        C().data.test_lmdb_dir,
        test_split,
        normalized_features_wts_file=C().data.norm_file,
        ignore=C().data.test_ignore,
        segment_len=C().data.segment_len,
        data_process_fn=process_foa_input_sed_doa_labels,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=C().data.batch_size,
        shuffle=False,
        num_workers=C().train.train_num_workers,
        collate_fn=test_dataset.collater,
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    logger.info(model)
    SetRandomSeed(C().train.seed)
    if C().model.pre_train:
        model.load_state_dict(torch.load(C().model.pre_train_model))

    optimizer = torch.optim.Adam(model.parameters(), lr=C().train.lr)
    total_steps = C().train.nb_steps
    warmup_steps = int(total_steps * 0.1)
    hold_steps = int(total_steps * 0.6)
    decay_steps = int(total_steps * 0.3)
    scheduler = TriStageLRScheduler(
        optimizer,
        peak_lr=C().train.lr,
        init_lr_scale=0.01,
        final_lr_scale=0.05,
        warmup_steps=warmup_steps,
        hold_steps=hold_steps,
        decay_steps=decay_steps,
    )
    epoch_count = 0
    step_count = 0

    # 开始训练
    stop_training = False
    best_seld_score = float("inf")  # 初始化最佳SELD分数
    best_epoch = 0  # 初始化最佳epoch
    best_checkpoint = ""  # 初始化最佳checkpoint路径
    patience = 40  # 早停耐心值
    patience_counter = 0  # 早停计数器
    while not stop_training:
        train_loss = []
        test_loss = []
        epoch_count += 1
        # 训练
        start_time = time.time()
        model.train()
        for data in train_dataloader:
            input = data["input"].to(device)
            target = data["target"].to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss.append(loss.item())
            step_count += 1
            if step_count % C().result.log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "epoch: {}, step: {}/{}, lr:{:.6f}, train_loss:{:.4f}".format(
                        epoch_count, step_count, total_steps, lr, loss.item()
                    )
                )
            if step_count >= total_steps:
                stop_training = True
                logger.info("Reached maximum number of steps")
                break

        torch.cuda.empty_cache()
        train_time = time.time() - start_time

        # 测试
        start_time = time.time()
        model.eval()
        test_result = SedDoaResult(segment_length=C().data.segment_len)
        for data in test_dataloader:
            input = data["input"].to(device)
            target = data["target"].to(device)
            with torch.no_grad():
                output = model(input)
                loss = criterion(output, target)
                test_loss.append(loss.item())

            test_result.add_items(data["wav_names"], output)
        output_dict = test_result.get_result()
        test_time = time.time() - start_time

        # 保存测试集CSV文件
        dcase_output_val_dir = os.path.join(
            C().result.dcase_output_dir,
            "epoch{}_step{}".format(epoch_count, step_count),
        )
        os.makedirs(dcase_output_val_dir, exist_ok=True)
        for csv_name, perfile_out_dict in output_dict.items():
            output_file = os.path.join(dcase_output_val_dir, "{}.csv".format(csv_name))
            write_output_format_file(output_file, perfile_out_dict)

        # 根据保存的CSV文件进行结果评估
        score_obj = ComputeSELDResults(ref_files_folder=C().data.ref_files_dir)
        val_ER, val_F, val_LE, val_LR, val_seld_scr, classwise_val_scr = (
            score_obj.get_SELD_Results(dcase_output_val_dir)
        )
        logger.info(
            "epoch: {}, step: {}/{}, train_time:{:.2f}, test_time:{:.2f}, average_train_loss:{:.4f}, average_test_loss:{:.4f}".format(
                epoch_count,
                step_count,
                total_steps,
                train_time,
                test_time,
                np.mean(train_loss),
                np.mean(test_loss),
            )
        )
        logger.info(
            "ER/F/LE/LR/SELD: {}".format(
                "{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}".format(
                    val_ER, val_F, val_LE, val_LR, val_seld_scr
                )
            )
        )
        # 保存模型
        checkpoint_output_dir = C().result.checkpoint_output_dir
        os.makedirs(checkpoint_output_dir, exist_ok=True)
        model_path = os.path.join(
            checkpoint_output_dir,
            "checkpoint_epoch{}_step{}.pth".format(epoch_count, step_count),
        )
        torch.save(model.state_dict(), model_path)
        logger.info("save checkpoint: {}".format(model_path))

        # 更新最佳性能记录
        if val_seld_scr < best_seld_score:
            best_seld_score = val_seld_scr
            best_epoch = epoch_count
            best_checkpoint = model_path
            patience_counter = 0  # 重置早停计数器
            logger.info(
                "New best model found SELD score: {:.4f}".format(best_seld_score)
            )
        else:
            patience_counter += 1
            logger.info(
                "No improvement for {} epochs. Best SELD score so far: {:.4f}".format(
                    patience_counter, best_seld_score
                )
            )

            # 检查是否应该早停
        if patience_counter >= patience:
            logger.info(
                "Early stopping triggered after {} epochs without improvement".format(
                    patience
                )
            )
            stop_training = True

    # 训练结束后记录最佳性能
    logger.info("=" * 50)
    logger.info("Training completed!")
    if patience_counter >= patience:
        logger.info("Stopped due to: Early stopping criterion met")
    else:
        logger.info("Stopped due to: Maximum steps reached")
    logger.info("Best performance:")
    logger.info("Epoch: {}".format(best_epoch))
    logger.info("SELD score: {:.4f}".format(best_seld_score))
    logger.info("Checkpoint path: {}".format(best_checkpoint))
    logger.info("Total epochs trained: {}".format(epoch_count))
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
