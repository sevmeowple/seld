import torch
from torch.utils.data import DataLoader

from config.config_manager import C, init_config, parse_cli_args
from seld.lr_scheduler.tri_stage_lr_scheduler import TriStageLRScheduler
from seld_v2.models.resnet_conformer import ResnetConformer
from seld.utils.process import SetRandomSeed

from seld_v2.data.dataset import LmdbDataset
from seld_v2.data.process import process_foa_input_sed_doa_labels
from seld_v2.losses.sed_doa_loss import SedDoaLoss
from seld_v2.metrics.result_collector import SedDoaResultCollector
from seld_v2.training.train_epoch import train_one_epoch
from seld_v2.training.eval_epoch import eval_one_epoch, save_and_evaluate
from seld_v2.training.checkpoint import EarlyStopping, save_checkpoint, load_checkpoint
from seld_v2.training.experiment import ExperimentDir


def main():
    cli_args = parse_cli_args()
    init_config(**cli_args)

    # 实验目录 & 日志
    exp = ExperimentDir(base_dir="experiments", name=C().exp_name)
    logger = exp.setup_logging()
    logger.info(C())
    logger.info("Experiment dir: %s", exp.root)

    # 模型 & 损失
    criterion = SedDoaLoss(loss_weight=[0.1, 1])
    model = ResnetConformer(
        in_channel=C().model.in_channel, in_dim=C().model.in_dim,
        out_dim=C().model.out_dim, att_context_size=C().model.att_context_size,
        num_conformer_layer=C().model.num_conformer_layers,
        encoder_dim=C().model.encoder_dim,
        use_dynamic_chunk=C().model.use_dynamic_chunk,
        chunk_candidates=C().model.chunk_candidates,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(model)
    SetRandomSeed(C().train.seed)
    if C().model.pre_train:
        load_checkpoint(model, C().model.pre_train_model)

    # 数据
    seg_len = int(C().data.segment_len)
    norm_file = C().data.norm_file
    train_dataset = LmdbDataset(
        C().data.train_lmdb_dir, [1, 2, 3], normalized_features_wts_file=norm_file,
        ignore=C().data.train_ignore, segment_len=seg_len, data_process_fn=process_foa_input_sed_doa_labels,
    )
    test_dataset = LmdbDataset(
        C().data.test_lmdb_dir, [4], normalized_features_wts_file=norm_file,
        ignore=C().data.test_ignore, segment_len=seg_len, data_process_fn=process_foa_input_sed_doa_labels,
    )
    batch_size = int(C().data.batch_size)
    num_workers = int(C().train.train_num_workers)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=train_dataset.collater)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=test_dataset.collater)

    # 优化器 & 调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=C().train.lr)
    total_steps = C().train.nb_steps
    scheduler = TriStageLRScheduler(
        optimizer, peak_lr=C().train.lr, init_lr_scale=0.01, final_lr_scale=0.05,
        warmup_steps=int(total_steps * 0.1),
        hold_steps=int(total_steps * 0.6),
        decay_steps=int(total_steps * 0.3),
    )

    # 训练循环
    step_count = 0
    early_stopping = EarlyStopping(patience=40)

    for epoch in range(1, 10000):
        # 训练
        train_result = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, step_count, total_steps, epoch, C().result.log_interval,
        )
        step_count = train_result["step_count"]

        # 验证
        collector = SedDoaResultCollector(segment_length=C().data.segment_len)
        eval_result = eval_one_epoch(model, test_loader, criterion, collector, device)

        logger.info(
            "epoch: %d, step: %d/%d, train_time:%.2f, test_time:%.2f, "
            "avg_train_loss:%.4f, avg_test_loss:%.4f",
            epoch, step_count, total_steps,
            train_result["train_time"], eval_result["eval_time"],
            train_result["train_loss"], eval_result["test_loss"],
        )

        # 评估 SELD 指标
        output_dir = exp.dcase_output_dir / f"epoch{epoch}_step{step_count}"
        seld_metrics = save_and_evaluate(eval_result["output_dict"], output_dir, C().data.ref_files_dir)

        # 保存检查点 & 早停
        ckpt_path = save_checkpoint(model, exp.checkpoint_dir, epoch, step_count)
        is_best = seld_metrics["seld_scr"] < early_stopping.best_score
        should_stop = early_stopping.step(seld_metrics["seld_scr"], epoch, ckpt_path)

        if is_best:
            exp.log_metrics("train", {
                "epoch": epoch, "step": step_count,
                "att_context_size": str(C().model.att_context_size),
                "use_dynamic_chunk": C().model.use_dynamic_chunk,
            }, seld_metrics)

        if train_result["stop_training"] or should_stop:
            if should_stop:
                logger.info("Early stopping triggered")
            break

    # 训练结束
    logger.info("=" * 50)
    logger.info("Training completed! Best epoch: %d, SELD: %.4f, ckpt: %s",
                early_stopping.best_epoch, early_stopping.best_score, early_stopping.best_checkpoint)


if __name__ == "__main__":
    main()
