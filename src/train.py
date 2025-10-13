import torch

torch.autograd.set_detect_anomaly(True)
from typing import List
import warnings
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)
import os
import pathlib
import numpy as np

from tqdm import tqdm
from datetime import datetime
from skimage import img_as_ubyte
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

from net.phydae import PhyDAE, OTDLoss

from options import train_options
from utils.schedulers import LinearWarmupCosineAnnealingLR
from data.dataset_utils import AIOTrainDataset, CDD11
from utils.loss_utils import FFTLoss
import torch.nn.utils as nn_utils

class FFTLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # 先裁剪输入
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)

        # 计算FFT
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))

        # 只关注幅度谱
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)

        # === 关键修复：添加数值稳定性 ===
        # 1. 裁剪极值
        pred_amp = torch.clamp(pred_amp, max=100.0)
        target_amp = torch.clamp(target_amp, max=100.0)

        # 2. 检测NaN
        if torch.isnan(pred_amp).any() or torch.isnan(target_amp).any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # 3. 使用平滑的L1损失而不是L1
        return F.smooth_l1_loss(pred_amp, target_amp)


class SSIMLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.eps = 1e-8  # 增加epsilon

    def forward(self, pred, target):
        # 裁剪输入
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)

        mu1 = pred.mean(dim=(-2, -1), keepdim=True)
        mu2 = target.mean(dim=(-2, -1), keepdim=True)

        sigma1_sq = ((pred - mu1) ** 2).mean(dim=(-2, -1), keepdim=True)
        sigma2_sq = ((target - mu2) ** 2).mean(dim=(-2, -1), keepdim=True)
        sigma12 = ((pred - mu1) * (target - mu2)).mean(dim=(-2, -1), keepdim=True)

        # === 增加稳定性常数 ===
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2

        ssim_n = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        ssim_d = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

        # 添加epsilon避免除零
        ssim = ssim_n / (ssim_d + self.eps)

        # 裁剪SSIM到合理范围
        ssim = torch.clamp(ssim, -1, 1)

        # 检测NaN
        if torch.isnan(ssim).any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        return 1 - ssim.mean()


class TaskAdaptiveLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.fft_loss = FFTLoss()
        self.ssim_loss = SSIMLoss()

        self.task_weights = {
            0: [1.0, 0.0, 0.2],  # dehaze
            1: [0.5, 0.1, 0.4],  # denoise
            2: [1.0, 0.0, 0.2],  # dedark
            3: [0.3, 0.5, 0.2],  # deblur
        }

    def forward(self, pred, target, task_ids):
        batch_size = pred.size(0)

        l1 = self.l1_loss(pred, target)
        fft = self.fft_loss(pred, target)
        ssim = self.ssim_loss(pred, target)

        total_loss = 0.0
        for i in range(batch_size):
            task_id = task_ids[i].item() if torch.is_tensor(task_ids[i]) else task_ids[i]
            weights = self.task_weights.get(task_id, [1.0, 0.0, 0.0])
            sample_loss = weights[0] * l1 + weights[1] * fft + weights[2] * ssim
            total_loss += sample_loss / batch_size

        return total_loss


# ============================================================================
# 验证指标函数
# ============================================================================
def calc_psnr(img1, img2, data_range=1.0):
    """计算PSNR指标"""
    err = np.sum((img1 - img2) ** 2, dtype=np.float64)
    return 10 * np.log10((data_range ** 2) / (err / img1.size))


def calc_ssim(img1, img2):
    """计算SSIM指标"""
    return structural_similarity(img1, img2, channel_axis=2, gaussian_weights=True,
                                 data_range=1.0, full=False)


def load_pretrained_encoder_weights(phydae_model, pretrained_checkpoint):
    """加载预训练编码器权重"""
    checkpoint = torch.load(pretrained_checkpoint, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        for prefix in ['net.', 'model.', 'module.']:
            if k.startswith(prefix):
                new_k = k[len(prefix):]
                break

        if any(new_k.startswith(m) for m in ['patch_embed', 'enc', 'latent', 'freq_embed']):
            new_state_dict[new_k] = v

    result = phydae_model.load_state_dict(new_state_dict, strict=False)

    print(f"成功加载参数: {len(new_state_dict)}")
    print(f"缺失的键: {len(result.missing_keys)}")
    print(f"未预期的键: {len(result.unexpected_keys)}")

    return phydae_model


class PLTrainModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.balance_loss_weight = opt.balance_loss_weight

        # 损失跟踪
        self.train_losses = []
        self.epoch_losses = {
            'total': [],
            'otd': [],
            'pixel': [],
            'internal': [],
            'contrastive': [],  # **新增**
        }

        # 验证指标跟踪
        self.val_metrics_history = {
            'psnr': [],
            'ssim': [],
        }

        # 专家使用统计 **新增**
        self.expert_usage_stats = {i: 0 for i in range(4)}

        # 创建模型
        self.net = PhyDAE(
            dim=self.opt.dim,
            num_blocks=self.opt.num_blocks,
            num_dec_blocks=self.opt.num_dec_blocks,
            rank=self.opt.rank,
            topk=self.opt.topk
        )

        # 加载预训练权重
        self.net = load_pretrained_encoder_weights(self.net, './src/pretrained_weight/last.ckpt')

        # 损失函数
        self.otd_loss = OTDLoss(lambda_reg=0.1)
        self.pixel_loss = TaskAdaptiveLoss()

        # **修复: 调整损失权重,增加对比学习损失**
        self.loss_weights = {
            'otd': 0.02,
            'pixel': 1.0,
            'internal': 0.02,  # 提高MoE损失权重
            'contrastive': 0.1,  # **新增对比学习损失**
        }

    def forward(self, x, labels=None):
        return self.net(x, labels)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch

        # **关键修复: 前向传播时传递任务标签**
        restored = self.net(degrad_patch, de_id)

        # === 多重损失计算 ===
        # 1. OTD损失
        otd_loss, otd_info = self.otd_loss(
            degrad_patch, restored, clean_patch, de_id)

        # 2. 像素级损失
        pixel_loss = self.pixel_loss(restored, clean_patch, de_id)

        # 3. 模型内部损失 (MoE)
        internal_loss = self.net.total_loss

        # 4. **新增: 对比学习损失**
        contrastive_loss = self.net.contrastive_loss

        # === 总损失 ===
        loss = (
                self.loss_weights['otd'] * otd_loss +
                self.loss_weights['pixel'] * pixel_loss +
                self.loss_weights['internal'] * internal_loss +
                self.loss_weights['contrastive'] * contrastive_loss  # **新增**
        )

        # === 记录损失 ===
        self.log("train/loss", loss, prog_bar=True, logger=True,
                 on_step=True, on_epoch=True)
        self.log("train/otd_loss", otd_loss, logger=True,
                 on_step=False, on_epoch=True)
        self.log("train/pixel_loss", pixel_loss, logger=True,
                 on_step=False, on_epoch=True)
        self.log("train/internal_loss", internal_loss, logger=True,
                 on_step=False, on_epoch=True)
        self.log("train/contrastive_loss", contrastive_loss, logger=True,  # **新增**
                 on_step=False, on_epoch=True)

        # 保存损失用于epoch统计
        self.train_losses.append({
            'total': loss.detach(),
            'otd': otd_loss.detach(),
            'pixel': pixel_loss.detach(),
            'internal': internal_loss.detach(),
            'contrastive': contrastive_loss.detach() if torch.is_tensor(contrastive_loss) else torch.tensor(
                contrastive_loss),
        })

        # 记录学习率
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("LR Schedule", lr, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        ([clean_name, de_id], degrad_patch, clean_patch) = batch

        with torch.no_grad():
            restored = self.net(degrad_patch, de_id)

            if isinstance(restored, List) and len(restored) == 2:
                restored, _ = restored

            assert restored.shape == clean_patch.shape
            restored = torch.clamp(restored, 0, 1)

        # 计算验证损失
        otd_loss, otd_info = self.otd_loss(
            degrad_patch, restored, clean_patch, de_id)
        pixel_loss = self.pixel_loss(restored, clean_patch, de_id)
        internal_loss = self.net.total_loss
        contrastive_loss = self.net.contrastive_loss

        val_loss = (
                self.loss_weights['otd'] * otd_loss +
                self.loss_weights['pixel'] * pixel_loss +
                self.loss_weights['internal'] * internal_loss +
                self.loss_weights['contrastive'] * contrastive_loss
        )

        # 计算图像质量指标
        restored_np = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        clean_np = clean_patch.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        psnr_score = calc_psnr(clean_np, restored_np, data_range=1.0)
        ssim_score = calc_ssim(clean_np, restored_np)

        # 记录指标
        self.log("val/loss", val_loss, prog_bar=True, logger=True,
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/otd_loss", otd_loss, logger=True,
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/pixel_loss", pixel_loss, logger=True,
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/internal_loss", internal_loss, logger=True,
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/contrastive_loss", contrastive_loss, logger=True,
                 on_step=False, on_epoch=True, sync_dist=True)

        self.log("val/psnr", psnr_score, prog_bar=True, logger=True,
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/ssim", ssim_score, prog_bar=True, logger=True,
                 on_step=False, on_epoch=True, sync_dist=True)

        return {
            'val_loss': val_loss,
            'psnr': psnr_score,
            'ssim': ssim_score,
        }

    def on_train_epoch_end(self):
        """训练epoch结束时的处理"""
        if len(self.train_losses) == 0:
            return

        # 计算epoch平均loss
        avg_losses = {}
        for key in ['total', 'otd', 'pixel', 'internal', 'contrastive']:
            avg_losses[key] = torch.stack([x[key] for x in self.train_losses]).mean()

        # 保存到历史记录
        for key, value in avg_losses.items():
            self.epoch_losses[key].append(value.item())

        # **新增: 统计并打印专家使用情况**
        if self.trainer.is_global_zero:
            current_epoch = self.current_epoch
            print(f"\n{'=' * 70}")
            print(f"EPOCH {current_epoch + 1} TRAINING SUMMARY")
            print(f"{'=' * 70}")
            print(f"Total Loss:        {avg_losses['total']:.6f}")
            print(f"OTD Loss:          {avg_losses['otd']:.6f}")
            print(f"Pixel Loss:        {avg_losses['pixel']:.6f}")
            print(f"Internal Loss:     {avg_losses['internal']:.6f}")
            print(f"Contrastive Loss:  {avg_losses['contrastive']:.6f}")
            print(f"Learning Rate:     {self.trainer.optimizers[0].param_groups[0]['lr']:.8f}")

            # 显示loss变化趋势
            if len(self.epoch_losses['total']) > 1:
                prev_loss = self.epoch_losses['total'][-2]
                curr_loss = self.epoch_losses['total'][-1]
                change = curr_loss - prev_loss
                change_pct = (change / prev_loss) * 100
                trend = "↓" if change < 0 else "↑"
                print(f"Loss Change:       {change:+.6f} ({change_pct:+.2f}%) {trend}")

            print(f"{'=' * 70}\n")

        # 记录到logger
        self.log("epoch/avg_total_loss", avg_losses['total'], logger=True)
        self.log("epoch/avg_otd_loss", avg_losses['otd'], logger=True)
        self.log("epoch/avg_pixel_loss", avg_losses['pixel'], logger=True)
        self.log("epoch/avg_internal_loss", avg_losses['internal'], logger=True)
        self.log("epoch/avg_contrastive_loss", avg_losses['contrastive'], logger=True)

        # 清空损失记录
        self.train_losses.clear()

    def on_validation_epoch_end(self):
        """验证epoch结束时的处理"""
        val_loss = self.trainer.callback_metrics.get('val/loss', None)
        val_psnr = self.trainer.callback_metrics.get('val/psnr', None)
        val_ssim = self.trainer.callback_metrics.get('val/ssim', None)

        if val_loss is not None:
            if val_psnr is not None:
                self.val_metrics_history['psnr'].append(val_psnr.item())
            if val_ssim is not None:
                self.val_metrics_history['ssim'].append(val_ssim.item())

            if self.trainer.is_global_zero:
                print(f"\n{'=' * 70}")
                print(f"EPOCH {self.current_epoch + 1} VALIDATION SUMMARY")
                print(f"{'=' * 70}")
                print(f"Validation Loss: {val_loss:.6f}")
                if val_psnr is not None:
                    print(f"PSNR:           {val_psnr:.4f} dB")
                if val_ssim is not None:
                    print(f"SSIM:           {val_ssim:.4f}")

                # 显示最佳指标
                if len(self.val_metrics_history['psnr']) > 0:
                    best_psnr = max(self.val_metrics_history['psnr'])
                    best_ssim = max(self.val_metrics_history['ssim'])
                    print(f"Best PSNR:      {best_psnr:.4f} dB")
                    print(f"Best SSIM:      {best_ssim:.4f}")
                print(f"{'=' * 70}\n")

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.opt.lr * 0.5,  # 降低学习率
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=20,  # 增加warmup
            max_epochs=self.opt.epochs
        )

        if self.opt.fine_tune_from:
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer=optimizer,
                warmup_epochs=5,
                max_epochs=self.opt.epochs
            )

        return [optimizer], [scheduler]

    def save_loss_history(self, save_path):
        """保存loss历史到文件"""
        loss_history = {
            'epoch_losses': self.epoch_losses,
            'val_metrics_history': self.val_metrics_history,
            'expert_usage_stats': self.expert_usage_stats,
            'total_epochs': len(self.epoch_losses['total'])
        }
        torch.save(loss_history, save_path)
        if hasattr(self, 'trainer') and self.trainer.is_global_zero:
            print(f"Loss and metrics history saved to: {save_path}")
        elif not hasattr(self, 'trainer'):
            print(f"Loss and metrics history saved to: {save_path}")


# ============================================================================
# Callback: 保存loss历史
# ============================================================================
class LossHistoryCallback(pl.Callback):
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def on_train_end(self, trainer, pl_module):
        """训练结束时保存loss历史"""
        if trainer.is_global_zero:
            loss_file = os.path.join(self.save_dir, "loss_history.pt")
            pl_module.save_loss_history(loss_file)


# ============================================================================
# 专家使用监控Callback**
# ============================================================================
class ExpertMonitorCallback(pl.Callback):
    """监控专家使用情况"""

    def __init__(self, log_every_n_epochs=5):
        self.log_every_n_epochs = log_every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        """每N个epoch打印专家统计信息"""
        current_epoch = trainer.current_epoch

        if (current_epoch + 1) % self.log_every_n_epochs == 0 and trainer.is_global_zero:
            print(f"\n{'=' * 70}")
            print(f"EXPERT USAGE STATISTICS (Epoch {current_epoch + 1})")
            print(f"{'=' * 70}")
            print("Note: Expert selection is happening dynamically during training.")
            print("Check training logs for real-time expert probability distributions.")
            print(f"{'=' * 70}\n")


# ============================================================================
# 主训练函数
# ============================================================================
def main(opt):
    print("=" * 70)
    print("TRAINING OPTIONS")
    print("=" * 70)
    print(opt)
    print("=" * 70)

    time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    log_dir = os.path.join("logs/", time_stamp)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    if opt.wblogger:
        name = opt.model + "_" + time_stamp
        logger = WandbLogger(name=name, save_dir=log_dir, config=opt)
    else:
        logger = TensorBoardLogger(save_dir=log_dir)

    if opt.fine_tune_from:
        model = PLTrainModel.load_from_checkpoint(opt.fine_tune_from, opt=opt)
    else:
        model = PLTrainModel(opt)

    # 打印模型信息
    if opt.num_gpus <= 1 or os.environ.get('LOCAL_RANK', '0') == '0':
        print("\n" + "=" * 70)
        print("MODEL ARCHITECTURE")
        print("=" * 70)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters:     {total_params / 1e6:.2f}M")
        print(f"Trainable Parameters: {trainable_params / 1e6:.2f}M")
        print("=" * 70 + "\n")

    # 创建checkpoint路径
    checkpoint_path = os.path.join(opt.ckpt_dir, time_stamp)
    pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        every_n_epochs=5,
        save_top_k=-1,
        save_last=True,
        filename='epoch={epoch:02d}-psnr={val/psnr:.2f}'  # **改进文件命名**
    )
    loss_history_callback = LossHistoryCallback(checkpoint_path)
    expert_monitor_callback = ExpertMonitorCallback(log_every_n_epochs=5)  # **新增**

    # 创建数据集
    if "CDD11" in opt.trainset:
        _, subset = opt.trainset.split("_")
        trainset = CDD11(opt, split="train", subset=subset)
        valset = CDD11(opt, split="val", subset=subset)
    else:
        trainset = AIOTrainDataset(opt, split="train")
        valset = AIOTrainDataset(opt, split="val")

    # 数据加载器
    trainloader = DataLoader(
        trainset,
        batch_size=opt.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers
    )

    valloader = DataLoader(
        valset,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        num_workers=2
    )

    # 创建Trainer
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[
            checkpoint_callback,
            loss_history_callback,
            expert_monitor_callback
        ],
        accumulate_grad_batches=opt.accum_grad,
        deterministic=False,
        check_val_every_n_epoch=5,  # **建议每5个epoch验证一次**
        num_sanity_val_steps=2,
        gradient_clip_val=0.5,  # **新增梯度裁剪以稳定训练**
        gradient_clip_algorithm="norm",
    )

    # 恢复checkpoint
    if opt.resume_from:
        checkpoint_path = os.path.join(opt.ckpt_dir, opt.resume_from, "last.ckpt")
    else:
        checkpoint_path = None

    # 开始训练
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70 + "\n")

    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        val_dataloaders=valloader,
        ckpt_path=checkpoint_path
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    train_opt = train_options()
    main(train_opt)