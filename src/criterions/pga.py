"""PGA-KD criterion: spectral geometry alignment (PGA) + mutual-information based
semantic consistency (SCL), on top of the usual InfoNCE + MSE baseline."""
import logging
from typing import Any, Dict, List, Optional, Tuple, Type
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import PreTrainedModel, ProcessorMixin
from ..arguments import SSAAguments
from src.model.llava.constants import IMAGE_TOKEN_INDEX

logger = logging.getLogger(__name__)

class SCLProjectors(nn.Module):
    """Maps teacher features into the student space, one head per SCL pathway."""
    def __init__(self, t_dim: int, s_dim: int):
        super().__init__()

        # intra-modal: teacher img/txt -> student img/txt
        self.img_proj = nn.Linear(t_dim, s_dim, bias=False)
        self.txt_proj = nn.Linear(t_dim, s_dim, bias=False)

        # inter-modal: teacher img -> student txt, and vice versa
        self.cross_img_proj = nn.Linear(t_dim, s_dim, bias=False)
        self.cross_txt_proj = nn.Linear(t_dim, s_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in [self.img_proj, self.txt_proj, self.cross_img_proj, self.cross_txt_proj]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')

    def forward(self, t_img: torch.Tensor, t_txt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (img_intra, txt_intra, img_cross, txt_cross)."""
        if t_img.dtype != self.img_proj.weight.dtype:
            self.img_proj.to(t_img.dtype)
            self.txt_proj.to(t_txt.dtype)
            self.cross_img_proj.to(t_img.dtype)
            self.cross_txt_proj.to(t_txt.dtype)
        t_img_intra = self.img_proj(t_img)
        t_txt_intra = self.txt_proj(t_txt)

        t_img_cross = self.cross_img_proj(t_img)
        t_txt_cross = self.cross_txt_proj(t_txt)

        return t_img_intra, t_txt_intra, t_img_cross, t_txt_cross


class PGAKDLoss(nn.Module):
    """L_total = L_Base + lambda_PGA * L_PGA + lambda_SCL * L_SCL"""

    def __init__(self, args: Type[SSAAguments]):
        super(PGAKDLoss, self).__init__()
        self.args = args
        # filled in on the first forward, once the distiller is available
        self.teacher_dim = None
        self.student_dim = None
        self.teacher_processor = None
        self.student_processor = None

        self.pga_weight = getattr(args, 'pga_loss_weight', getattr(args, 'kd_weight', 1.0))
        self.scl_weight = getattr(args, 'scl_loss_weight', 1.0)
        self.mse_weight = getattr(args, 'mse_loss_weight', 0.3)
        self.projector_lr = getattr(args, 'projector_lr', 5e-4)
        self.variance_threshold = getattr(args, 'pga_spectral_variance_threshold', 0.95)

        self.mse_loss_fn = nn.MSELoss(reduction='mean')

        self.scl_projectors = None
        self.initialized_scl_projectors = False

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0

        logger.info(f"PGAKDLoss initialized | PGA W: {self.pga_weight} | SCL W: {self.scl_weight} | MSE W: {self.mse_weight}")

    def _dist_gather_tensor(self, t: torch.Tensor) -> torch.Tensor:
        """Gathers tensors from all GPUs, keeping the local one differentiable."""
        if self.world_size <= 1:
            return t
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        return torch.cat(all_tensors, dim=0)

    def forward(self, distiller: nn.Module, input_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        student_model: PreTrainedModel = distiller.student
        teacher_model: PreTrainedModel = distiller.teacher

        if not self.initialized_scl_projectors:
            self.teacher_dim = distiller.teacher_hidden_dim
            self.student_dim = distiller.student_hidden_dim
            self.teacher_processor = distiller.get_teacher_processor()
            self.student_processor = distiller.get_student_processor()

        # 1. Encode query/positive with both models
        with torch.no_grad():
            teacher_model.eval()
            t_qry, _, t_qry_attn, t_qry_hiddens = teacher_model.encode_input(input_data['teacher_inputs']['qry'])
            t_pos, _, _, t_pos_hiddens = teacher_model.encode_input(input_data['teacher_inputs']['pos'])

        s_qry, _, s_qry_attn, s_qry_hiddens = student_model.encode_input(input_data['student_inputs']['qry'])
        s_pos, _, _, s_pos_hiddens = student_model.encode_input(input_data['student_inputs']['pos'])

        # 2. Contrastive loss over the global batch
        if self.world_size > 1:
            all_s_qry = self._dist_gather_tensor(s_qry)
            all_s_pos = self._dist_gather_tensor(s_pos)
        else:
            all_s_qry = s_qry
            all_s_pos = s_pos

        scores = student_model.compute_similarity(all_s_qry, all_s_pos)
        scores = scores.view(all_s_qry.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_s_pos.size(0) // all_s_qry.size(0))

        loss_infonce = nn.CrossEntropyLoss()(scores / distiller.temperature, target)

        # 3. Point-wise regression against the projected teacher embedding
        loss_mse = torch.tensor(0.0, device=s_qry.device)
        if hasattr(distiller, "projectors") and "t2s_txt" in distiller.projectors:
            projected_t_qry = distiller.projectors["t2s_txt"](t_qry)
            loss_mse = self.mse_loss_fn(s_qry, projected_t_qry)

        # 4. Geometry and semantics
        loss_pga = self._compute_pga_loss(s_qry, s_pos, t_qry, t_pos)

        loss_scl = self._compute_scl_loss(
            s_qry_hiddens, t_qry_hiddens,
            s_qry_attn, t_qry_attn,
            input_data['student_inputs']['qry']['input_ids'],
            input_data['teacher_inputs']['qry']['input_ids'],
            self.student_processor,
            self.teacher_processor,
            distiller.temperature
        )

        total_loss = loss_infonce + \
                     (self.mse_weight * loss_mse) + \
                     (self.pga_weight * loss_pga) + \
                     (self.scl_weight * loss_scl)

        return {
            "loss": total_loss,
            "contrastive_loss": loss_infonce,
            "mse_loss": loss_mse,
            "pga_loss": loss_pga,
            "scl_loss": loss_scl
        }

    # ---- PGA ----

    def _compute_pga_loss(self,
                          s_qry: torch.Tensor, s_pos: torch.Tensor,
                          t_qry: torch.Tensor, t_pos: torch.Tensor) -> torch.Tensor:
        """CKA between the student geometry and the denoised teacher geometry."""
        s_batch = torch.cat([s_qry, s_pos], dim=0)
        t_batch = torch.cat([t_qry, t_pos], dim=0)

        K_T = torch.matmul(t_batch, t_batch.t())
        K_S = torch.matmul(s_batch, s_batch.t())

        K_T_Clean = self._spectral_filter(K_T, self.variance_threshold)

        return 1.0 - self._centered_kernel_alignment(K_S, K_T_Clean)

    def _spectral_filter(self, gram_matrix: torch.Tensor, threshold: float) -> torch.Tensor:
        """Rebuilds the Gram matrix from the top eigenvectors holding `threshold` of the energy."""
        orig_dtype = gram_matrix.dtype
        gram_matrix_fp32 = gram_matrix.to(torch.float32)

        L, V = torch.linalg.eigh(gram_matrix_fp32)
        L = L.flip(0)
        V = V.flip(1)
        L = torch.clamp(L, min=0.0)

        total_variance = torch.sum(L)
        if total_variance <= 1e-8:
            return gram_matrix

        cumulative_variance = torch.cumsum(L, dim=0) / total_variance
        k = torch.searchsorted(cumulative_variance, threshold) + 1
        k = min(k.item(), len(L))

        L_k = torch.diag(L[:k])
        V_k = V[:, :k]
        K_reconstructed = V_k @ L_k @ V_k.t()

        return K_reconstructed.to(orig_dtype)

    def _centered_kernel_alignment(self, K_S: torch.Tensor, K_T: torch.Tensor) -> torch.Tensor:
        """Linear CKA; scale-invariant, so the student is not pushed to match eigenvalue magnitudes."""
        orig_dtype = K_S.dtype
        K_S = K_S.to(torch.float32)
        K_T = K_T.to(torch.float32)

        n = K_S.size(0)
        unit = torch.ones([n, n], device=K_S.device)
        I = torch.eye(n, device=K_S.device)
        H = I - unit / n

        K_S_c = torch.matmul(torch.matmul(H, K_S), H)
        K_T_c = torch.matmul(torch.matmul(H, K_T), H)

        hsic = torch.trace(torch.matmul(K_S_c, K_T_c))
        var_s = torch.sqrt(torch.trace(torch.matmul(K_S_c, K_S_c)))
        var_t = torch.sqrt(torch.trace(torch.matmul(K_T_c, K_T_c)))

        cka_score = hsic / (var_s * var_t + 1e-8)
        return cka_score.to(orig_dtype)

    # ---- SCL ----

    def _compute_scl_loss(self,
                          s_hidden: Any, t_hidden: Any,
                          s_attn: Any, t_attn: Any,
                          s_ids: torch.Tensor, t_ids: torch.Tensor,
                          processor_student: Optional[ProcessorMixin],
                          processor_teacher: Optional[ProcessorMixin],
                          temperature: float) -> torch.Tensor:
        """MI maximization over the two intra- and two inter-modal pathways.

        `*_hidden` are the per-layer hidden states from `encode_input` (index 3) and
        `*_attn` the per-layer attention matrices (index 2).
        """
        if processor_student is None or processor_teacher is None:
            logger.warning("SCL loss requires processors for both student and teacher models.")
            return torch.tensor(0.0, device=s_ids.device)

        if not self.initialized_scl_projectors:
            self.scl_projectors = SCLProjectors(self.teacher_dim, self.student_dim).to(s_hidden[-1].device)
            self.initialized_scl_projectors = True

        s_img, s_txt = self._get_image_text_representation(s_hidden, s_attn, processor_student, s_ids)
        t_img_raw, t_txt_raw = self._get_image_text_representation(t_hidden, t_attn, processor_teacher, t_ids)

        t_img_intra, t_txt_intra, t_img_cross, t_txt_cross = self.scl_projectors(t_img_raw, t_txt_raw)

        loss_img_img = self._info_nce(s_img, t_img_intra, temperature)
        loss_txt_txt = self._info_nce(s_txt, t_txt_intra, temperature)
        loss_img_txt = self._info_nce(s_img, t_txt_cross, temperature)
        loss_txt_img = self._info_nce(s_txt, t_img_cross, temperature)

        return (loss_img_img + loss_txt_txt + loss_img_txt + loss_txt_img) / 4.0

    def _info_nce(self, s_features: torch.Tensor, t_features: torch.Tensor, temp: float) -> torch.Tensor:
        s_norm = F.normalize(s_features, dim=-1)
        t_norm = F.normalize(t_features, dim=-1)

        if self.world_size > 1:
            s_norm = self._dist_gather_tensor(s_norm)
            t_norm = self._dist_gather_tensor(t_norm)

        logits = torch.matmul(s_norm, t_norm.T) / temp
        labels = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)

        return nn.CrossEntropyLoss()(logits, labels)

    def _get_image_text_representation(self, hidden_states: Any, attentions: Any, processor: ProcessorMixin, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attention-pools one image and one text vector per sample.

        The image placeholder is expanded into many tokens inside the model, so the input mask
        no longer lines up with the hidden states and has to be rebuilt against the output length.
        """
        last_hidden_state = hidden_states[-1] # (B, Seq_out, D)
        device = last_hidden_state.device

        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, device=device)

        # image token id: from the processor if exposed, otherwise the LLaVA/FastVLM constant
        img_token_id = None
        if hasattr(processor, 'image_token_id'):
            img_token_id = processor.image_token_id
        elif hasattr(processor, 'tokenizer'):
             token_str = getattr(processor, 'image_token', '<image>')
             img_token_id = processor.tokenizer.convert_tokens_to_ids(token_str)

        if img_token_id is None:
             try:
                 img_token_id = IMAGE_TOKEN_INDEX
             except ImportError:
                 logger.warning("Could not determine image_token_id. Using EOS rep fallback.")
                 eos = last_hidden_state[:, -1, :]
                 return eos, eos

        batch_size, seq_len_in = input_ids.shape
        _, seq_len_out, _ = last_hidden_state.shape

        # one placeholder was replaced by num_img_tokens, so the length grew by num_img_tokens - 1
        expansion_diff = seq_len_out - seq_len_in
        num_img_tokens = expansion_diff + 1 if expansion_diff >= 0 else 0

        # text-only batch: treat the placeholder itself as the single image token
        if expansion_diff == 0:
            num_img_tokens = 1

        expanded_image_mask = torch.zeros((batch_size, seq_len_out), dtype=torch.bool, device=device)
        expanded_text_mask = torch.zeros((batch_size, seq_len_out), dtype=torch.bool, device=device)

        pad_token_id = getattr(processor.tokenizer, 'pad_token_id', -100)

        # the image sits at a different offset in every sample, so map the masks per sample
        for i in range(batch_size):
            img_indices = (input_ids[i] == img_token_id).nonzero(as_tuple=True)[0]

            if len(img_indices) == 0:
                # no image: everything that is not padding is text
                is_pad = (input_ids[i] == pad_token_id)
                expanded_text_mask[i, :seq_len_in] = ~is_pad
            else:
                # VLM2Vec samples carry at most one image
                start_idx = img_indices[0].item()

                img_end_idx = start_idx + num_img_tokens
                expanded_image_mask[i, start_idx : img_end_idx] = True

                # text before the image keeps its position, text after it shifts by expansion_diff
                text_pre_mask = (input_ids[i, :start_idx] != pad_token_id)
                expanded_text_mask[i, :start_idx] = text_pre_mask

                text_post_mask = (input_ids[i, start_idx+1:] != pad_token_id)
                if len(text_post_mask) > 0:
                    expanded_text_mask[i, img_end_idx:] = text_post_mask

        # pool with the last token's attention, averaged over heads
        last_attention = attentions[-1]
        last_token_attention = last_attention[:, :, -1, :]
        avg_attn = last_token_attention.mean(dim=1) # (B, Seq_out)

        def get_weighted_rep(mask):
            has_tokens = mask.sum(dim=1) > 0

            # -inf outside the modality so softmax renormalizes within it
            masked_attn = avg_attn.masked_fill(~mask, torch.finfo(avg_attn.dtype).min)
            normalized_weights = torch.softmax(masked_attn, dim=-1).unsqueeze(1)

            weighted_rep = torch.bmm(normalized_weights, last_hidden_state).squeeze(1)

            # fall back to EOS when the modality is absent (e.g. image rep of a text-only sample)
            eos_rep = last_hidden_state[:, -1, :]
            return torch.where(has_tokens.unsqueeze(-1), weighted_rep, eos_rep)

        image_representation = get_weighted_rep(expanded_image_mask)
        text_representation = get_weighted_rep(expanded_text_mask)

        return image_representation, text_representation

    def _add_optimizer_param_group(self, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        lr = self.projector_lr
        optimizer.add_param_group({
            "params": self.scl_projectors.parameters(),
            "lr": lr
        })
        logger.info(f"SCL Projector parameters added to optimizer with lr: {lr}")
        return optimizer
