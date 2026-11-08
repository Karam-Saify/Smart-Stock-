"""Shared long-only exposure caps and drawdown/loss cooldown."""
import numpy as np

def clip_weights(w, limits):
    w = np.clip(np.nan_to_num(w, nan=0.0), 0.0, limits.max_weight)
    total = w.sum()
    return w * (limits.max_gross / total) if total > limits.max_gross else w

class RiskGovernor:
    def __init__(self, limits):
        self.limits, self.peak, self.flat_until, self.events = limits, 1.0, -1, []

    def update(self, step, equity, day_return=0.0):
        if step < self.flat_until:
            return True
        if self.flat_until >= 0 and step == self.flat_until:
            self.peak, self.flat_until = equity, -1
        self.peak = max(self.peak, equity)
        dd = 1 - equity / self.peak
        if dd >= self.limits.kill_drawdown:
            self.flat_until = step + self.limits.kill_cooldown_days
            self.events.append((step, "drawdown_kill", float(dd)))
            return True
        if day_return <= -self.limits.daily_loss_limit:
            self.flat_until = step + self.limits.daily_loss_cooldown_days
            self.events.append((step, "daily_loss_halt", float(day_return)))
            return True
        return False
