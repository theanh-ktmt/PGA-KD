import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from scipy.optimize import linear_sum_assignment

class EMKDLoss(nn.Module):
    def __init__(self, args):
        super(EMKDLoss, self).__init__()
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss()
        self.kd_loss_weight = self.args.kd_weight
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0
            
    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors
    
    def forward(self, distiller, input_data):
        self.distiller = distiller
        student_model = distiller.student
        teacher_model = distiller.teacher
        
        student_qry_input = input_data['student_inputs']['qry']
        student_pos_input = input_data['student_inputs']['pos']
        
        teacher_qry_input = input_data['teacher_inputs']['qry']
        teacher_pos_input = input_data['teacher_inputs']['pos']
        # Original Teacher counts
        num_text_qry_tokens = ((teacher_qry_input['input_ids'] < 151643) | (teacher_qry_input['input_ids'] > 151656)).sum(dim=1)
        num_text_pos_tokens = ((teacher_pos_input['input_ids'] < 151643) | (teacher_pos_input['input_ids'] > 151656)).sum(dim=1)
        
        # student counts, assuming it shares the tokenizer/image ids
        num_text_qry_tokens_stu = ((student_qry_input['input_ids'] < 151643) | (student_qry_input['input_ids'] > 151656)).sum(dim=1)
        num_text_pos_tokens_stu = ((student_pos_input['input_ids'] < 151643) | (student_pos_input['input_ids'] > 151656)).sum(dim=1)
        
        batch_size = student_qry_input['input_ids'].size(0)
        with torch.no_grad():
            teacher_model.eval()
            teacher_qry_output = teacher_model.encode_input(teacher_qry_input)
            teacher_pos_output = teacher_model.encode_input(teacher_pos_input)
            teacher_qry_reps, teacher_qry_image_features, teacher_qry_attention, teacher_qry_hidden_states = teacher_qry_output
            teacher_pos_reps, teacher_pos_image_features, teacher_pos_attention, teacher_pos_hidden_states = teacher_pos_output
        
        student_qry_output = student_model.encode_input(student_qry_input)
        student_pos_output = student_model.encode_input(student_pos_input)
        student_qry_reps, student_qry_image_features, student_qry_attention, student_qry_hidden_states = student_qry_output
        student_pos_reps, student_pos_image_features, student_pos_attention, student_pos_hidden_states = student_pos_output
        
        if self.world_size > 1:
            all_student_qry_reps = self._dist_gather_tensor(student_qry_reps)
            all_student_pos_reps = self._dist_gather_tensor(student_pos_reps)
        else:
            all_student_qry_reps = student_qry_reps
            all_student_pos_reps = student_pos_reps
            
        scores = student_model.compute_similarity(all_student_qry_reps, all_student_pos_reps)
        scores = scores.view(all_student_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_student_qry_reps.size(0) // all_student_pos_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / self.distiller.temperature, target)
        
        loss_distill = 0.0
        cur_idx_qry_img = 0
        cur_idx_pos_img = 0
        loss_vsd = 0.0
        loss_vlad = 0.0
        # Compute KD Loss
        for i in range(batch_size):
            if student_qry_image_features is not None and teacher_qry_image_features is not None:
                if cur_idx_qry_img < len(student_qry_image_features) and cur_idx_qry_img < len(teacher_qry_image_features):
                    if student_qry_image_features[cur_idx_qry_img] is not None and teacher_qry_image_features[cur_idx_qry_img] is not None:
                        num_tokens_vision_qry_stu = student_qry_image_features[cur_idx_qry_img].size(0)
                        num_tokens_vision_qry_tea = teacher_qry_image_features[cur_idx_qry_img].size(0)
                        num_text_token_qry_tea = num_text_qry_tokens[i]
                        num_text_token_qry_stu = num_text_qry_tokens_stu[i]
                        
                        # slice the student the same way as the teacher
                        student_qry_vision_hidden_state = student_qry_hidden_states[-1][i][-(num_tokens_vision_qry_stu + num_text_token_qry_stu):-(num_text_token_qry_stu), :]
                        teacher_qry_vision_hidden_state = teacher_qry_hidden_states[-1][i][-(num_tokens_vision_qry_tea + num_text_token_qry_tea):-(num_text_token_qry_tea), :]
                        
                        student_qry_text_hidden_state = student_qry_hidden_states[-1][i][-num_text_token_qry_stu:, :]
                        teacher_qry_text_hidden_state = teacher_qry_hidden_states[-1][i][-num_text_token_qry_tea:, :]
                        vl_s = student_model.encoder.lm_head(student_qry_vision_hidden_state)
                        vl_t = teacher_model.encoder.lm_head(teacher_qry_vision_hidden_state)
                        
                        c_match = torch.sum(
                            torch.abs(vl_t.unsqueeze(1) - vl_s.unsqueeze(0)),
                            dim=-1
                        ).float().detach().cpu().numpy()
                        
                        idx_t, idx_s = linear_sum_assignment(c_match)
                        idx_t = torch.tensor(idx_t, dtype=torch.long, device=vl_t.device)
                        idx_s = torch.tensor(idx_s, dtype=torch.long, device=vl_s.device)
                        
                        # skip when no vision tokens matched
                        if len(idx_s) > 0:
                            vl_s_matched = vl_s[idx_s]
                            vl_t_matched = vl_t[idx_t]
                            loss_vsd += nn.MSELoss()(vl_s_matched, vl_t_matched)
                            
                            vhs_s_matched = student_qry_vision_hidden_state[idx_s]
                            vhs_t_matched = teacher_qry_vision_hidden_state[idx_t]
                            
                            min_text_len_qry = min(student_qry_text_hidden_state.size(0), teacher_qry_text_hidden_state.size(0))
                            
                            # VLAD needs at least one text token
                            if min_text_len_qry > 0:
                                student_qry_text_hidden_state = student_qry_text_hidden_state[-min_text_len_qry:, :]
                                teacher_qry_text_hidden_state = teacher_qry_text_hidden_state[-min_text_len_qry:, :]
                                
                                affinity_s = F.cosine_similarity(vhs_s_matched.unsqueeze(1), student_qry_text_hidden_state.unsqueeze(0), dim=-1)
                                affinity_t = F.cosine_similarity(vhs_t_matched.unsqueeze(1), teacher_qry_text_hidden_state.unsqueeze(0), dim=-1)
                                
                                loss_vlad += F.smooth_l1_loss(affinity_s, affinity_t)
                        cur_idx_qry_img += 1
                        
            if student_pos_image_features is not None and teacher_pos_image_features is not None:
                if cur_idx_pos_img < len(student_pos_image_features) and cur_idx_pos_img < len(teacher_pos_image_features):
                    if student_pos_image_features[cur_idx_pos_img] is not None and teacher_pos_image_features[cur_idx_pos_img] is not None:
                        num_tokens_vision_pos_stu = student_pos_image_features[cur_idx_pos_img].size(0)
                        num_tokens_vision_pos_tea = teacher_pos_image_features[cur_idx_pos_img].size(0)
                        num_text_token_pos_tea = num_text_pos_tokens[i]
                        num_text_token_pos_stu = num_text_pos_tokens_stu[i]
                        student_pos_vision_hidden_state = student_pos_hidden_states[-1][i][-(num_tokens_vision_pos_stu + num_text_token_pos_stu):-(num_text_token_pos_stu), :]
                        teacher_pos_vision_hidden_state = teacher_pos_hidden_states[-1][i][-(num_tokens_vision_pos_tea + num_text_token_pos_tea):-(num_text_token_pos_tea), :]
                        
                        student_pos_text_hidden_state = student_pos_hidden_states[-1][i][-num_text_token_pos_stu:, :]
                        teacher_pos_text_hidden_state = teacher_pos_hidden_states[-1][i][-num_text_token_pos_tea:, :]
                        vp_s = student_model.encoder.lm_head(student_pos_vision_hidden_state)
                        vp_t = teacher_model.encoder.lm_head(teacher_pos_vision_hidden_state)
                        
                        c_match = torch.sum(
                            torch.abs(vp_t.unsqueeze(1) - vp_s.unsqueeze(0)),
                            dim=-1
                        ).float().detach().cpu().numpy()
                        
                        idx_t, idx_s = linear_sum_assignment(c_match)
                        idx_t = torch.tensor(idx_t, dtype=torch.long, device=vp_t.device)
                        idx_s = torch.tensor(idx_s, dtype=torch.long, device=vp_s.device)
                        
                        # skip when no vision tokens matched
                        if len(idx_s) > 0:
                            vp_s_matched = vp_s[idx_s]
                            vp_t_matched = vp_t[idx_t]
                            
                            loss_vsd += nn.MSELoss()(vp_s_matched, vp_t_matched)
                            
                            vhs_s_matched = student_pos_vision_hidden_state[idx_s]
                            vhs_t_matched = teacher_pos_vision_hidden_state[idx_t]
                            
                            min_text_len_pos = min(student_pos_text_hidden_state.size(0), teacher_pos_text_hidden_state.size(0))
                            
                            # VLAD needs at least one text token
                            if min_text_len_pos > 0:
                                student_pos_text_hidden_state = student_pos_text_hidden_state[-min_text_len_pos:, :]
                                teacher_pos_text_hidden_state = teacher_pos_text_hidden_state[-min_text_len_pos:, :]
                                
                                affinity_s = F.cosine_similarity(vhs_s_matched.unsqueeze(1), student_pos_text_hidden_state.unsqueeze(0), dim=-1)
                                affinity_t = F.cosine_similarity(vhs_t_matched.unsqueeze(1), teacher_pos_text_hidden_state.unsqueeze(0), dim=-1)
                                
                                loss_vlad += F.smooth_l1_loss(affinity_s, affinity_t)
                        cur_idx_pos_img += 1
                        
        loss_distill = (0.25 * loss_vsd + 25 * loss_vlad) / batch_size
        
        loss_mse = 0.5* (nn.MSELoss()(student_qry_reps, self.distiller.projectors["t2s"](teacher_qry_reps)) + nn.MSELoss()(student_pos_reps, self.distiller.projectors["t2s"](teacher_pos_reps)))
        loss = 0.5 * contrastive_loss + 0.5 * loss_mse + loss_distill
        return {
            'loss': loss,
            'contrastive_loss': contrastive_loss,
            'kd_loss': loss_distill
        }
                        