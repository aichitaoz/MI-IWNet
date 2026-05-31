"""
Early Stopping机制

用于在验证集性能不再提升时提前停止训练，避免过拟合
"""

import numpy as np


class EarlyStopping:
    """
    Early Stopping类

    监控验证集指标，在连续patience个epoch没有改善时停止训练

    Args:
        patience: 容忍的epoch数量（默认10）
        min_delta: 最小改善量，小于此值不算改善（默认0.001）
        mode: 'min'表示越小越好（如loss），'max'表示越大越好（如IoU）
        verbose: 是否打印详细信息

    使用示例：
        early_stopping = EarlyStopping(patience=10, mode='min')

        for epoch in range(max_epochs):
            train_loss = train(...)
            val_loss = validate(...)

            if early_stopping(val_loss):
                print(f"Early stopping triggered at epoch {epoch}")
                break
    """
    def __init__(self, patience=10, min_delta=0.001, mode='min', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

        # 根据mode设置比较函数
        if mode == 'min':
            self.is_better = lambda a, b: a < b - min_delta
        elif mode == 'max':
            self.is_better = lambda a, b: a > b + min_delta
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def __call__(self, score, epoch=None):
        """
        检查是否应该早停

        Args:
            score: 当前epoch的验证指标
            epoch: 当前epoch数（用于打印）

        Returns:
            bool: True表示应该早停，False表示继续训练
        """
        if self.best_score is None:
            # 第一次调用
            self.best_score = score
            self.best_epoch = epoch if epoch is not None else 0
            if self.verbose:
                print(f"📌 Early Stopping initialized: best score = {score:.6f}")
            return False

        if self.is_better(score, self.best_score):
            # 性能改善
            if self.verbose:
                improvement = abs(score - self.best_score)
                print(f"✅ Validation improved: {self.best_score:.6f} -> {score:.6f} (+{improvement:.6f})")

            self.best_score = score
            self.best_epoch = epoch if epoch is not None else 0
            self.counter = 0
            return False
        else:
            # 性能没有改善
            self.counter += 1

            if self.verbose:
                print(f"⏸️  No improvement for {self.counter}/{self.patience} epochs "
                      f"(best: {self.best_score:.6f} at epoch {self.best_epoch})")

            if self.counter >= self.patience:
                if self.verbose:
                    print(f"🛑 Early Stopping triggered!")
                    print(f"   Best score: {self.best_score:.6f} at epoch {self.best_epoch}")
                    print(f"   No improvement for {self.patience} consecutive epochs")
                self.early_stop = True
                return True

            return False

    def reset(self):
        """重置Early Stopping状态"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def state_dict(self):
        """返回状态字典（用于保存checkpoint）"""
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'mode': self.mode
        }

    def load_state_dict(self, state_dict):
        """从状态字典恢复（用于恢复训练）"""
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.best_epoch = state_dict['best_epoch']
        self.patience = state_dict.get('patience', self.patience)
        self.min_delta = state_dict.get('min_delta', self.min_delta)
        self.mode = state_dict.get('mode', self.mode)


