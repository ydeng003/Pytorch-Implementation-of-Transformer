#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 19:31:28 2019

@author: cathy
"""

import numpy as np

class OptimScheduler():
    """
    A simple wrapper class for learning rate schedulingv
    
    """
    def __init__(self, optimizer, d_model=512, warmup_steps=4000):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.step_num = 0
        self.d_model = d_model

    def step_and_update_lr(self):
        #Step with the inner optimizer
        self._update_lr()
        self.optimizer.step()

    def zero_grad(self):
        #Zero out the gradients by the inner optimizer
        self.optimizer.zero_grad()

    def _update_lr(self):
        #Learning rate scheduling per step

        self.step_num = self.step_num + 1
        # increasing the learning rate linearly for the ﬁrst warmup_steps training steps
        # decreasing it thereafter proportionally to the inverse square root of the step number.
        lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.step_num, -0.5),
            self.step_num * np.power(self.warmup_steps, -1.5)])
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
