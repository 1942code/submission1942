import torch
import torch.nn as nn
from random import sample
import numpy as np
import torch.nn.functional as F

class MoCo(nn.Module):
    def __init__(self, base_encoder, config):
        super(MoCo, self).__init__()
        
        # 编码器
        self.encoder_q = base_encoder(input_dim=config['input_dim'], num_class=config['num_class'], low_dim=config['low_dim'])
        # 动量编码器
        self.encoder_k = base_encoder(input_dim=config['input_dim'], num_class=config['num_class'], low_dim=config['low_dim'])

        # 复制参数并冻结动量编码器
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # 初始化
            param_k.requires_grad = False  # 不通过梯度更新

        # 创建队列
        self.register_buffer("queue", torch.randn(config['low_dim'], config['moco_queue']))        
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))        
        self.register_buffer("prototypes", torch.zeros(config['num_class'], config['low_dim']))

    @torch.no_grad()
    def _momentum_update_key_encoder(self, config):
        """更新动量编码器"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * config['moco_m'] + param_q.data * (1. - config['moco_m'])

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, config):
        """队列更新机制"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert config['moco_queue'] % batch_size == 0  # 简化处理

        # 在ptr位置替换keys（出队和入队）
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % config['moco_queue']  # 移动指针
        self.queue_ptr[0] = ptr
    

    
    def forward(self, img, target, config, is_eval=False, is_proto=False):
        # 获取编码器输出
        output, q, u, x_q = self.encoder_q(img)
        
        if is_eval:  
            return output, q, target
        
        # 计算增强特征
        img_aug = img.clone().cuda()
        with torch.no_grad():
            self._momentum_update_key_encoder(config)
            _, k, _, _ = self.encoder_k(img_aug)
            
        # 计算实例对比损失
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= config['temperature']
        
        # 更新队列
        self._dequeue_and_enqueue(k, config) 
        
        if is_proto:     
            # 计算原型对比损失
            prototypes = self.prototypes.clone().detach()
            logits_proto = torch.mm(q, prototypes.t())/config['temperature']        
        else:
            logits_proto = 0
            
        # 选择置信样本
        confident_features, confident_targets = self._select_confident_samples(q, output, target)
        
        # 更新类原型（仅使用置信样本）
        if len(confident_features) > 0:
            for feat, label in zip(confident_features, confident_targets):
                # 使用动量更新机制更新原型，mp=0.999
                self.prototypes[label] = self.prototypes[label] * 0.999 + (1-0.999) * feat

        # 标准化原型
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)        

        return output, target, logits, x_q, logits_proto, u
