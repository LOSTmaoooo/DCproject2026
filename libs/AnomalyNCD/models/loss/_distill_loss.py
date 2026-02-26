import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class DistillLoss(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs, num_labeled_classes, num_unlabeled_classes,
                ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                student_temp=0.1, repeat_times=1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops

        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, int(warmup_teacher_temp_epochs / repeat_times)).repeat(repeat_times),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))        
        self.num_labeled_classes = num_labeled_classes
        self.num_unlabeled_classes = num_unlabeled_classes

    def forward(self, student_output, teacher_output, epoch, sample_weights):                                      
        """                                                                                        
        Cross-entropy between softmax outputs of the teacher and student networks.
        Args:
            student_output: [torch.Tensor]. Student network output.
            teacher_output: [torch.Tensor]. Teacher network output.
            epoch: [int]. Current epoch.
            sample_weights: [torch.Tensor]. Sample weights.
        Returns:
            total_loss: [torch.Tensor]. Total loss.
        """
        
        student_out = student_output / self.student_temp                                                                
        student_out = student_out.chunk(self.ncrops)                                               

        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)                                    

        # Set the value of the first self.num_labeled_classes class to 0 in teacher_output
        teacher_out[:, :self.num_labeled_classes] = 0
        teacher_out[:, self.num_labeled_classes:] /= teacher_out[:, self.num_labeled_classes:].sum(dim=-1, keepdim=True)
        
        # set the first class in unlabeled classes as normal label.
        normal_output = torch.zeros_like(teacher_out).cuda(non_blocking=True)
        normal_output[:, self.num_labeled_classes] = 1

        # Pseudo Labels Correction
        # transform the anomaly score to sample weight and adjust the pseudo labels from teacher output
        teacher_out = normal_output * sample_weights.unsqueeze(1) + teacher_out * (1 - sample_weights.unsqueeze(1))
        
        teacher_out = teacher_out.detach().chunk(2)                                                

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms


        return total_loss