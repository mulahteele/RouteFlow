"""Route Autoencoder: Encoder + Decoder wrapper with training and inference."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rdkit import Chem

from routeflow.data.route import Route, RouteStep, RouteToken, TokenType
from routeflow.models.embeddings import RouteEmbedding
from routeflow.models.encoder import RouteEncoder
from routeflow.models.decoder import RouteDecoder


class RouteAutoencoder(nn.Module):
    """Full autoencoder: encode route → z, decode z → route.

    Training: teacher forcing with parallel computation.
    Inference: autoregressive with closed-loop RDKit execution.
    """

    def __init__(
        self,
        num_rxn: int,
        bb_fingerprints: np.ndarray,  # (num_bb, fp_nbits)
        embed_dim: int = 256,
        latent_dim: int = 128,
        encoder_num_layers: int = 4,
        encoder_nhead: int = 8,
        encoder_ff_dim: int = 1024,
        encoder_dropout: float = 0.1,
        decoder_num_layers: int = 6,
        decoder_nhead: int = 8,
        decoder_ff_dim: int = 1024,
        decoder_dropout: float = 0.1,
        max_len: int = 64,
        fp_nbits: int = 2048,
    ):
        super().__init__()
        self.num_rxn = num_rxn
        self.num_bb = bb_fingerprints.shape[0]
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.fp_nbits = fp_nbits

        # Shared embedding layer
        self.embedding = RouteEmbedding(
            embed_dim=embed_dim,
            num_rxn_templates=num_rxn,
            bb_fingerprints=bb_fingerprints,
            fp_nbits=fp_nbits,
        )

        # Encoder
        self.encoder = RouteEncoder(
            embed_dim=embed_dim,
            latent_dim=latent_dim,
            num_layers=encoder_num_layers,
            nhead=encoder_nhead,
            ff_dim=encoder_ff_dim,
            dropout=encoder_dropout,
            max_len=max_len,
        )

        # Decoder
        self.decoder = RouteDecoder(
            embed_dim=embed_dim,
            latent_dim=latent_dim,
            num_rxn=num_rxn,
            num_layers=decoder_num_layers,
            nhead=decoder_nhead,
            ff_dim=decoder_ff_dim,
            dropout=decoder_dropout,
            max_len=max_len,
        )

    def encode(
        self,
        token_types: torch.Tensor,
        token_values: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode route → z. Returns (B, latent_dim)."""
        emb = self.embedding(token_types, token_values)
        z = self.encoder(emb, padding_mask)
        return z

    def decode_teacher_forcing(
        self,
        z: torch.Tensor,
        token_types: torch.Tensor,
        token_values: torch.Tensor,
        padding_mask: torch.Tensor,
        scheduled_sampling_p: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Decode with teacher forcing + optional scheduled sampling.

        Args:
            scheduled_sampling_p: probability of using model's own prediction
                instead of ground truth at each position. 0.0 = pure teacher forcing,
                0.5 = 50% chance of using model prediction.

        Source logits are NOT computed here (would be 200K wide).
        Source loss is computed per-position in compute_loss() using masked candidates.
        """
        # Shift: input is all tokens except last, target is all tokens except first
        inp_types = token_types[:, :-1]
        inp_values = token_values[:, :-1]
        inp_mask = padding_mask[:, :-1]

        # Get ground truth embeddings
        emb = self.embedding(inp_types, inp_values)

        if scheduled_sampling_p > 0 and self.training:
            # Two-pass scheduled sampling:
            # Pass 1: run with ground truth to get predictions at all positions
            with torch.no_grad():
                h_tf = self.decoder(emb, z, inp_mask)
                pred_action = self.decoder.predict_action(h_tf).argmax(dim=-1)  # (B, L-1)
                pred_rxn = self.decoder.predict_rxn(h_tf).argmax(dim=-1)        # (B, L-1)

            # Build predicted embeddings for action and rxn positions
            # (BB/POP positions keep ground truth — too expensive to predict)
            tgt_types = token_types[:, 1:]  # target types for each position
            B, L_minus1 = tgt_types.shape

            # Sample which positions to replace (only action and rxn positions)
            replace_mask = torch.rand(B, L_minus1, device=emb.device) < scheduled_sampling_p

            # For action positions: replace with predicted action embedding
            # action_id: 0=BRANCH, 1=REACT, 2=END
            is_action_tgt = (tgt_types == TokenType.BRANCH) | (tgt_types == TokenType.REACT) | (tgt_types == TokenType.END)
            action_replace = replace_mask & is_action_tgt
            if action_replace.any():
                # Map action_id to embedding index: BRANCH→0, REACT→1, END uses special_emb(1)
                pred_act_flat = pred_action[action_replace]  # (N,)
                pred_emb_flat = torch.zeros(pred_act_flat.shape[0], self.embed_dim, device=emb.device)
                # BRANCH (action_id=0) → action_emb(0)
                mask_br = pred_act_flat == 0
                if mask_br.any():
                    pred_emb_flat[mask_br] = self.embedding.action_emb(torch.zeros(mask_br.sum(), dtype=torch.long, device=emb.device))
                # REACT (action_id=1) → action_emb(1)
                mask_re = pred_act_flat == 1
                if mask_re.any():
                    pred_emb_flat[mask_re] = self.embedding.action_emb(torch.ones(mask_re.sum(), dtype=torch.long, device=emb.device))
                # END (action_id=2) → special_emb(1)
                mask_end = pred_act_flat == 2
                if mask_end.any():
                    pred_emb_flat[mask_end] = self.embedding.special_emb(torch.ones(mask_end.sum(), dtype=torch.long, device=emb.device))
                emb[action_replace] = pred_emb_flat

            # For rxn positions: replace with predicted rxn embedding
            is_rxn_tgt = tgt_types == TokenType.RXN
            rxn_replace = replace_mask & is_rxn_tgt
            if rxn_replace.any():
                pred_rxn_flat = pred_rxn[rxn_replace].clamp(0, self.num_rxn - 1)
                emb[rxn_replace] = self.embedding.rxn_emb(pred_rxn_flat)

        # Pass 2 (or only pass if no scheduled sampling): run decoder
        h = self.decoder(emb, z, inp_mask)  # (B, L-1, embed_dim)

        # Only compute action and rxn logits (cheap: 3 and 115 classes)
        action_logits = self.decoder.predict_action(h)  # (B, L-1, 3)
        rxn_logits = self.decoder.predict_rxn(h)        # (B, L-1, num_rxn)

        return {
            "action_logits": action_logits,
            "rxn_logits": rxn_logits,
            "hidden": h,  # (B, L-1, embed_dim) — source loss computed from this
        }

    def _compute_source_loss_per_position(
        self,
        h: torch.Tensor,              # (B, L-1, embed_dim)
        tgt_types: torch.Tensor,       # (B, L-1)
        tgt_values: torch.Tensor,      # (B, L-1)
        tgt_mask: torch.Tensor,        # (B, L-1) True=pad
        source_candidates: list[list[dict]] | None = None,
        # source_candidates[b][k] = {"candidates": np.array of BB indices, "is_pop_valid": bool}
        # for the k-th source position in batch element b
    ) -> torch.Tensor:
        """Compute source CE loss using per-position masked candidates (~2K).

        For each source position (BB or POP target), we:
        1. Get the compat-masked BB candidates + Pop option
        2. Compute dot product only with those candidates
        3. CE loss against the ground-truth index in this small set
        """
        device = h.device
        B, L_minus1, D = h.shape

        # Get the source head's MLP and pop head
        source_mlp = self.decoder.source_head.mlp
        pop_head = self.decoder.source_head.pop_head

        # BB repr matrix (full, on device)
        bb_repr_full = self.embedding.get_bb_repr_matrix()  # (num_bb, embed_dim)

        # Find all source positions
        is_source = (
            (tgt_types == TokenType.BB) | (tgt_types == TokenType.POP)
        ) & (~tgt_mask)

        if not is_source.any():
            return torch.tensor(0.0, device=device)

        # Collect losses per source position
        losses = []

        for b in range(B):
            b_source_mask = is_source[b]  # (L-1,)
            if not b_source_mask.any():
                continue

            source_indices = torch.where(b_source_mask)[0]  # positions in sequence

            for k, pos in enumerate(source_indices):
                h_pos = h[b, pos]  # (embed_dim,)

                # Determine candidates for this position
                if source_candidates is not None and b < len(source_candidates):
                    src_info = source_candidates[b]
                    if k < len(src_info):
                        cand_bb_ids = src_info[k]["candidates"]  # np.array of BB indices
                        is_pop_valid = src_info[k]["is_pop_valid"]
                    else:
                        # Fallback: use ground truth BB only + random negatives
                        cand_bb_ids, is_pop_valid = self._fallback_candidates(
                            tgt_types[b, pos], tgt_values[b, pos]
                        )
                else:
                    cand_bb_ids, is_pop_valid = self._fallback_candidates(
                        tgt_types[b, pos], tgt_values[b, pos]
                    )

                # Build candidate set: [Pop, BB_c0, BB_c1, ...]
                cand_ids_tensor = torch.from_numpy(cand_bb_ids).long().to(device)
                cand_repr = bb_repr_full[cand_ids_tensor]  # (n_cand, embed_dim)

                # Compute query
                query = source_mlp(h_pos)   # (embed_dim,)
                bb_logits = query @ cand_repr.T  # (n_cand,)

                # Pop logit
                pop_logit = pop_head(h_pos)  # (1,)

                if is_pop_valid:
                    # logits = [pop_logit, bb_logit_0, bb_logit_1, ...]
                    all_logits = torch.cat([pop_logit, bb_logits], dim=0)  # (1+n_cand,)
                else:
                    # logits = [bb_logit_0, bb_logit_1, ...], no Pop
                    all_logits = bb_logits  # (n_cand,)

                # Determine target index in this candidate set
                if tgt_types[b, pos] == TokenType.POP:
                    if is_pop_valid:
                        target_idx = 0  # Pop is at index 0
                    else:
                        continue  # shouldn't happen, skip
                else:
                    gt_bb_id = tgt_values[b, pos].item()
                    # Find position of gt_bb_id in candidate set
                    match = (cand_ids_tensor == gt_bb_id).nonzero(as_tuple=True)[0]
                    if len(match) == 0:
                        continue  # ground truth not in candidates, skip
                    offset = 1 if is_pop_valid else 0
                    target_idx = match[0].item() + offset

                target_tensor = torch.tensor(target_idx, dtype=torch.long, device=device)
                loss = F.cross_entropy(all_logits.unsqueeze(0), target_tensor.unsqueeze(0))
                losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=device)

        return torch.stack(losses).mean()

    def _fallback_candidates(self, tgt_type, tgt_value, n_neg=255):
        """Fallback when no compat info: use ground truth + random negatives."""
        import numpy as np
        if tgt_type == TokenType.POP:
            # Just a few random BBs + Pop
            cand = np.random.randint(0, self.num_bb, size=n_neg)
            return cand, True
        else:
            gt_bb = tgt_value.item()
            neg = np.random.randint(0, self.num_bb, size=n_neg)
            cand = np.unique(np.concatenate([[gt_bb], neg]))
            return cand, False

    def compute_loss(
        self,
        logits: dict[str, torch.Tensor],
        token_types: torch.Tensor,
        token_values: torch.Tensor,
        padding_mask: torch.Tensor,
        source_candidates: list[list[dict]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute CE loss for action, reaction, and source predictions.

        Source loss uses per-position masked candidates (~2K) instead of full 200K softmax.
        """
        tgt_types = token_types[:, 1:]
        tgt_values = token_values[:, 1:]
        tgt_mask = padding_mask[:, 1:]

        B, L_minus1 = tgt_types.shape
        device = tgt_types.device

        action_logits = logits["action_logits"]
        rxn_logits = logits["rxn_logits"]
        h = logits["hidden"]

        # --- Action loss ---
        is_action_pos = (
            (tgt_types == TokenType.BRANCH)
            | (tgt_types == TokenType.REACT)
            | (tgt_types == TokenType.END)
        ) & (~tgt_mask)

        action_targets = torch.zeros(B, L_minus1, dtype=torch.long, device=device)
        action_targets[tgt_types == TokenType.BRANCH] = 0
        action_targets[tgt_types == TokenType.REACT] = 1
        action_targets[tgt_types == TokenType.END] = 2

        loss_action = torch.tensor(0.0, device=device)
        if is_action_pos.any():
            loss_action = F.cross_entropy(
                action_logits[is_action_pos],
                action_targets[is_action_pos],
            )

        # --- Reaction loss ---
        is_rxn_pos = (tgt_types == TokenType.RXN) & (~tgt_mask)
        rxn_targets = tgt_values.clone()
        rxn_targets[~is_rxn_pos] = 0

        loss_rxn = torch.tensor(0.0, device=device)
        if is_rxn_pos.any():
            loss_rxn = F.cross_entropy(
                rxn_logits[is_rxn_pos],
                rxn_targets[is_rxn_pos],
            )

        # --- Source loss (per-position masked candidates) ---
        loss_source = self._compute_source_loss_per_position(
            h, tgt_types, tgt_values, tgt_mask, source_candidates
        )

        total_loss = loss_action + loss_rxn + loss_source

        return {
            "loss": total_loss,
            "loss_action": loss_action,
            "loss_rxn": loss_rxn,
            "loss_source": loss_source,
        }

    def forward(self, batch: dict, noise_sigma: float = 0.0,
                scheduled_sampling_p: float = 0.0) -> dict:
        """Full forward pass: encode → (optional noise) → decode (teacher forcing) → loss.

        Args:
            noise_sigma: if > 0, inject Gaussian noise into z before decoding.
            scheduled_sampling_p: probability of replacing ground truth with model
                prediction at each position (0.0 = pure teacher forcing).
        """
        token_types = batch["token_types"]
        token_values = batch["token_values"]
        padding_mask = batch["padding_mask"]
        source_candidates = batch.get("source_candidates", None)

        # Encode
        z = self.encode(token_types, token_values, padding_mask)

        # Latent noise injection (training only)
        z_dec = z
        if noise_sigma > 0 and self.training:
            z_dec = z + torch.randn_like(z) * noise_sigma

        # Decode with teacher forcing + scheduled sampling
        logits = self.decode_teacher_forcing(
            z_dec, token_types, token_values, padding_mask,
            scheduled_sampling_p=scheduled_sampling_p,
        )

        # Compute loss (source loss uses per-position masked candidates)
        losses = self.compute_loss(
            logits, token_types, token_values, padding_mask, source_candidates
        )
        losses["z"] = z

        return losses

    # ================================================================
    #  Closed-loop autoregressive decoding (inference)
    # ================================================================

    @torch.no_grad()
    def decode_autoregressive(
        self,
        z: torch.Tensor,                       # (1, latent_dim) single z
        compat_matrix,                          # CompatibilityMatrix
        bb_mols: list,                          # list of RDKit Mol (num_bb,)
        max_steps: int = 64,
        temperature: float = 1.0,
        greedy: bool = False,
    ) -> Optional[Route]:
        """Decode z into a complete route via autoregressive closed-loop generation.

        Args:
            greedy: if True, use argmax instead of sampling (for reconstruction eval)

        At each step:
        1. Predict next token type (action / rxn / source)
        2. Apply appropriate mask (action mask, BU RXN mask, compat BB mask, Pop mask)
        3. Greedy argmax or sample from masked distribution
        4. If reaction step complete, execute via RDKit, push to stack
        5. Inject mol_state token with fp(product)

        Returns Route or None if decoding fails.
        """
        from routeflow.chem.executor import SynthesisStack, ReactionExecutor

        device = z.device
        self.eval()

        stack = SynthesisStack()
        route = Route()

        # BB repr matrix for source head (precompute once)
        bb_repr = self.embedding.get_bb_repr_matrix()  # (num_bb, embed_dim)

        # Start with [START] token
        token_types_list = [TokenType.START.value]
        token_values_list = [0]

        # State machine
        # After START, expect: action (BRANCH / REACT / END)
        expect = "action"
        current_rxn_idx = None
        current_action = None
        current_rxn_obj = None
        remaining_positions = []
        collected_reactants = []  # list of RDKit Mol for current step

        for step in range(max_steps):
            # Build input tensors from current sequence
            seq_len = len(token_types_list)
            tt = torch.tensor([token_types_list], dtype=torch.long, device=device)   # (1, L)
            tv = torch.tensor([token_values_list], dtype=torch.long, device=device)  # (1, L)
            pad = torch.zeros(1, seq_len, dtype=torch.bool, device=device)           # (1, L)

            # Get embeddings and run decoder
            emb = self.embedding(tt, tv)
            h = self.decoder(emb, z, pad)          # (1, L, embed_dim)
            h_last = h[0, -1]                       # (embed_dim,) last position

            if expect == "action":
                # Predict action: BRANCH(0) / REACT(1) / END(2)
                logits = self.decoder.predict_action(h_last)  # (3,)

                # Action mask. END is *not* masked here — the model is free to
                # predict it at any point. Validity of an END prediction is
                # checked when it actually gets emitted (see below).
                action_mask = torch.tensor([True, True, True], device=device)  # [BR, RE, END]
                if stack.is_empty:
                    # Must Branch (no molecule to React on)
                    action_mask[1] = False  # no React
                else:
                    # Check if React is possible (BU RXN mask non-empty)
                    bu_rxns = compat_matrix.get_bu_rxn_mask(stack.top())
                    if not bu_rxns:
                        action_mask[1] = False  # no React possible

                logits[~action_mask] = -float("inf")
                if greedy:
                    action_id = logits.argmax().item()
                else:
                    probs = F.softmax(logits / temperature, dim=-1)
                    action_id = torch.multinomial(probs, 1).item()

                if action_id == 2:  # END
                    # Validate: stack must have exactly one molecule (closed
                    # loop) AND the previous token must be BB or POP (last
                    # step emitted a source — not a uni-mol React ending).
                    # If either fails, the model's END is invalid → drop the
                    # route so it counts as a decode failure in the metrics.
                    if stack.depth != 1:
                        return None
                    if token_types_list[-1] not in (
                        TokenType.BB.value, TokenType.POP.value,
                    ):
                        return None
                    route.tokens.append(RouteToken(TokenType.END))
                    break
                elif action_id == 0:  # BRANCH
                    current_action = "branch"
                    token_types_list.append(TokenType.BRANCH.value)
                    token_values_list.append(0)
                    route.tokens.append(RouteToken(TokenType.BRANCH))
                    expect = "rxn"
                else:  # REACT
                    current_action = "react"
                    token_types_list.append(TokenType.REACT.value)
                    token_values_list.append(0)
                    route.tokens.append(RouteToken(TokenType.REACT))
                    expect = "rxn"

            elif expect == "rxn":
                # Predict reaction
                logits = self.decoder.predict_rxn(h_last)  # (num_rxn,)

                if current_action == "branch":
                    # No mask for Branch — all 115 reactions allowed
                    # But filter to seed reactions (all positions fillable by BBs)
                    rxn_mask = torch.zeros(self.num_rxn, dtype=torch.bool, device=device)
                    seed_rxns = compat_matrix.get_seed_reactions()
                    rxn_mask[seed_rxns] = True
                else:
                    # React: BU RXN mask on stack top
                    rxn_mask = torch.zeros(self.num_rxn, dtype=torch.bool, device=device)
                    bu_rxns = compat_matrix.get_bu_rxn_mask(stack.top())
                    for r in bu_rxns:
                        rxn_mask[r] = True

                if not rxn_mask.any():
                    return None  # no valid reaction

                logits[~rxn_mask] = -float("inf")
                if greedy:
                    rxn_idx = logits.argmax().item()
                else:
                    probs = F.softmax(logits / temperature, dim=-1)
                    rxn_idx = torch.multinomial(probs, 1).item()

                current_rxn_idx = rxn_idx
                current_rxn_obj = compat_matrix.reactions[rxn_idx]
                token_types_list.append(TokenType.RXN.value)
                token_values_list.append(rxn_idx)
                route.tokens.append(RouteToken(TokenType.RXN, rxn_idx))

                # Determine remaining positions to fill
                n_reactants = current_rxn_obj.num_reactants
                if current_action == "branch":
                    # All positions need BBs
                    remaining_positions = list(range(n_reactants))
                    collected_reactants = [None] * n_reactants
                else:
                    # React: stack top is always assigned to slot 0
                    # (BU rxn mask already restricted to reactions where slot 0
                    # accepts stack top, matching enumerate_routes).
                    mol_top = stack.top()
                    if not current_rxn_obj.reactant_templates[0].match(mol_top):
                        return None
                    pos0 = 0
                    collected_reactants = [None] * n_reactants
                    collected_reactants[pos0] = mol_top
                    remaining_positions = [p for p in range(n_reactants) if p != pos0]

                if remaining_positions:
                    expect = "source"
                else:
                    # Uni-molecular react: no remaining positions
                    expect = "execute"

            elif expect == "source":
                # Predict source for next remaining position
                pos = remaining_positions[0]

                bb_logits, pop_logit = self.decoder.predict_source(h_last, bb_repr)
                # bb_logits: (num_bb,), pop_logit: (1,)

                # Build source logits: [Pop, BB_0, BB_1, ...]
                source_logits = torch.cat([pop_logit, bb_logits], dim=-1)  # (1+num_bb,)

                # Source mask
                source_mask = torch.zeros(1 + self.num_bb, dtype=torch.bool, device=device)

                # BB mask from compatibility
                compat_bbs = compat_matrix.get_compatible_bbs(current_rxn_idx, pos)
                if len(compat_bbs) > 0:
                    source_mask[compat_bbs + 1] = True  # +1 because index 0 is Pop

                # Pop mask
                can_pop = (
                    current_action == "react"
                    and not stack.is_empty
                    and current_rxn_obj.reactant_templates[pos].match(stack.top())
                )
                if can_pop:
                    source_mask[0] = True

                if not source_mask.any():
                    return None  # no valid source

                source_logits[~source_mask] = -float("inf")
                if greedy:
                    source_id = source_logits.argmax().item()
                else:
                    probs = F.softmax(source_logits / temperature, dim=-1)
                    source_id = torch.multinomial(probs, 1).item()

                if source_id == 0:
                    # Pop
                    pop_mol, _ = stack.pop()
                    collected_reactants[pos] = pop_mol
                    token_types_list.append(TokenType.POP.value)
                    token_values_list.append(0)
                    route.tokens.append(RouteToken(TokenType.POP))
                else:
                    # BB
                    bb_idx = source_id - 1
                    collected_reactants[pos] = bb_mols[bb_idx]
                    token_types_list.append(TokenType.BB.value)
                    token_values_list.append(bb_idx)
                    route.tokens.append(RouteToken(TokenType.BB, bb_idx))

                remaining_positions.pop(0)

                if remaining_positions:
                    expect = "source"  # more positions to fill
                else:
                    expect = "execute"

            if expect == "execute":
                # Execute reaction via RDKit
                if current_action == "react":
                    if stack.is_empty:
                        return None  # stack was consumed by Pop sources
                    stack.pop()  # remove stack top (pos0) — already in collected_reactants

                product = ReactionExecutor.execute(
                    current_rxn_obj, collected_reactants
                )
                if product is None:
                    return None  # reaction failed

                product_smi = Chem.MolToSmiles(product, canonical=True)
                if ReactionExecutor.num_heavy_atoms(product) > 200:
                    return None  # too large

                # Push product to stack
                stack.push(product, product_smi)

                # Record step
                sources = []
                for pos in range(current_rxn_obj.num_reactants):
                    if current_action == "react" and pos == 0:
                        continue  # pos0 was implicit pop
                    # Check what was placed at this position
                    # (simplified: look back in tokens)
                step = RouteStep(
                    action=current_action,
                    rxn_idx=current_rxn_idx,
                    sources=sources,
                    product_smiles=product_smi,
                )
                route.steps.append(step)
                route.intermediates.append(product_smi)

                # Reset for next step
                current_rxn_idx = None
                current_action = None
                current_rxn_obj = None
                remaining_positions = []
                collected_reactants = []
                expect = "action"

        # Finalize. The loop only breaks here if the model successfully
        # predicted (and we validated) an END token. If we fall through —
        # i.e. max_steps was hit without a valid END — the route is invalid.
        if not (route.tokens and route.tokens[-1].token_type == TokenType.END):
            return None

        route.num_reactions = len(route.steps)
        route.tree_depth = 0  # approximate
        if stack.is_empty:
            return None
        route.final_product = stack.top_smiles()
        return route

    @torch.no_grad()
    def decode_beam_search(
        self,
        z: torch.Tensor,                       # (1, latent_dim) single z
        compat_matrix,                          # CompatibilityMatrix
        bb_mols: list,                          # list of RDKit Mol (num_bb,)
        beam_width: int = 5,
        max_steps: int = 64,
    ) -> Optional[Route]:
        """Decode z using beam search over the autoregressive state machine.

        Maintains beam_width independent decoding states. At each prediction
        step (action, rxn, source), expands each beam by top-k candidates and
        prunes to beam_width total by cumulative log probability.

        Returns the highest-probability completed route, or None if all fail.
        """
        from routeflow.chem.executor import SynthesisStack, ReactionExecutor
        import copy as copy_module

        device = z.device
        self.eval()

        bb_repr = self.embedding.get_bb_repr_matrix()

        def make_initial_beam():
            return {
                "token_types": [TokenType.START.value],
                "token_values": [0],
                "stack": SynthesisStack(),
                "route": Route(),
                "expect": "action",
                "current_rxn_idx": None,
                "current_action": None,
                "current_rxn_obj": None,
                "remaining_positions": [],
                "collected_reactants": [],
                "log_prob": 0.0,
                "done": False,
            }

        def clone_beam(b):
            return {
                "token_types": list(b["token_types"]),
                "token_values": list(b["token_values"]),
                "stack": b["stack"].copy(),
                "route": copy_module.deepcopy(b["route"]),
                "expect": b["expect"],
                "current_rxn_idx": b["current_rxn_idx"],
                "current_action": b["current_action"],
                "current_rxn_obj": b["current_rxn_obj"],
                "remaining_positions": list(b["remaining_positions"]),
                "collected_reactants": list(b["collected_reactants"]),
                "log_prob": b["log_prob"],
                "done": False,
            }

        def try_execute(nb):
            """Try to execute reaction for a beam. Returns True on success, False on failure."""
            if nb["current_action"] == "react":
                if nb["stack"].is_empty:
                    return False
                nb["stack"].pop()

            product = ReactionExecutor.execute(nb["current_rxn_obj"], nb["collected_reactants"])
            if product is None:
                return False

            product_smi = Chem.MolToSmiles(product, canonical=True)
            if ReactionExecutor.num_heavy_atoms(product) > 80:
                return False

            nb["stack"].push(product, product_smi)
            nb["route"].steps.append(RouteStep(
                action=nb["current_action"], rxn_idx=nb["current_rxn_idx"],
                sources=[], product_smiles=product_smi,
            ))
            nb["route"].intermediates.append(product_smi)
            nb["current_rxn_idx"] = None
            nb["current_action"] = None
            nb["current_rxn_obj"] = None
            nb["remaining_positions"] = []
            nb["collected_reactants"] = []
            nb["expect"] = "action"
            return True

        active_beams = [make_initial_beam()]
        completed = []

        for step in range(max_steps):
            if not active_beams:
                break

            next_beams = []

            for beam in active_beams:
                # Build input tensors
                tt = torch.tensor([beam["token_types"]], dtype=torch.long, device=device)
                tv = torch.tensor([beam["token_values"]], dtype=torch.long, device=device)
                pad = torch.zeros(1, len(beam["token_types"]), dtype=torch.bool, device=device)

                emb = self.embedding(tt, tv)
                h = self.decoder(emb, z, pad)
                h_last = h[0, -1]

                if beam["expect"] == "action":
                    logits = self.decoder.predict_action(h_last)
                    # END is *not* masked — model predicts it freely; validity
                    # is checked when END is actually emitted (see below).
                    action_mask = torch.tensor([True, True, True], device=device)
                    if beam["stack"].is_empty:
                        action_mask[1] = False
                    else:
                        bu_rxns = compat_matrix.get_bu_rxn_mask(beam["stack"].top())
                        if not bu_rxns:
                            action_mask[1] = False

                    logits[~action_mask] = -float("inf")
                    log_probs = F.log_softmax(logits, dim=-1)

                    for idx in torch.where(action_mask)[0]:
                        action_id = idx.item()
                        nb = clone_beam(beam)
                        nb["log_prob"] += log_probs[action_id].item()

                        if action_id == 2:  # END
                            # Validate END: stack must have exactly 1 molecule
                            # AND prev token must be BB or POP. If invalid,
                            # drop this beam.
                            if nb["stack"].depth != 1:
                                continue
                            if nb["token_types"][-1] not in (
                                TokenType.BB.value, TokenType.POP.value,
                            ):
                                continue
                            nb["route"].tokens.append(RouteToken(TokenType.END))
                            nb["done"] = True
                            nb["route"].num_reactions = len(nb["route"].steps)
                            nb["route"].final_product = nb["stack"].top_smiles()
                            completed.append(nb)
                        elif action_id == 0:  # BRANCH
                            nb["current_action"] = "branch"
                            nb["token_types"].append(TokenType.BRANCH.value)
                            nb["token_values"].append(0)
                            nb["route"].tokens.append(RouteToken(TokenType.BRANCH))
                            nb["expect"] = "rxn"
                            next_beams.append(nb)
                        else:  # REACT
                            nb["current_action"] = "react"
                            nb["token_types"].append(TokenType.REACT.value)
                            nb["token_values"].append(0)
                            nb["route"].tokens.append(RouteToken(TokenType.REACT))
                            nb["expect"] = "rxn"
                            next_beams.append(nb)

                elif beam["expect"] == "rxn":
                    logits = self.decoder.predict_rxn(h_last)
                    if beam["current_action"] == "branch":
                        rxn_mask = torch.zeros(self.num_rxn, dtype=torch.bool, device=device)
                        rxn_mask[compat_matrix.get_seed_reactions()] = True
                    else:
                        rxn_mask = torch.zeros(self.num_rxn, dtype=torch.bool, device=device)
                        for r in compat_matrix.get_bu_rxn_mask(beam["stack"].top()):
                            rxn_mask[r] = True

                    if not rxn_mask.any():
                        continue

                    logits[~rxn_mask] = -float("inf")
                    log_probs = F.log_softmax(logits, dim=-1)
                    topk = min(beam_width, int(rxn_mask.sum().item()))
                    top_vals, top_idxs = log_probs.topk(topk)

                    for val, idx in zip(top_vals, top_idxs):
                        rxn_idx = idx.item()
                        nb = clone_beam(beam)
                        nb["log_prob"] += val.item()
                        nb["current_rxn_idx"] = rxn_idx
                        nb["current_rxn_obj"] = compat_matrix.reactions[rxn_idx]
                        nb["token_types"].append(TokenType.RXN.value)
                        nb["token_values"].append(rxn_idx)
                        nb["route"].tokens.append(RouteToken(TokenType.RXN, rxn_idx))

                        n_reactants = nb["current_rxn_obj"].num_reactants
                        if nb["current_action"] == "branch":
                            nb["remaining_positions"] = list(range(n_reactants))
                            nb["collected_reactants"] = [None] * n_reactants
                        else:
                            mol_top = nb["stack"].top()
                            if not nb["current_rxn_obj"].reactant_templates[0].match(mol_top):
                                continue
                            pos0 = 0
                            nb["collected_reactants"] = [None] * n_reactants
                            nb["collected_reactants"][pos0] = mol_top
                            nb["remaining_positions"] = [p for p in range(n_reactants) if p != pos0]

                        if nb["remaining_positions"]:
                            nb["expect"] = "source"
                        else:
                            nb["expect"] = "execute"
                        next_beams.append(nb)

                elif beam["expect"] == "source":
                    pos = beam["remaining_positions"][0]
                    bb_logits, pop_logit = self.decoder.predict_source(h_last, bb_repr)
                    source_logits = torch.cat([pop_logit, bb_logits], dim=-1)

                    source_mask = torch.zeros(1 + self.num_bb, dtype=torch.bool, device=device)
                    compat_bbs = compat_matrix.get_compatible_bbs(beam["current_rxn_idx"], pos)
                    if len(compat_bbs) > 0:
                        source_mask[compat_bbs + 1] = True
                    can_pop = (
                        beam["current_action"] == "react"
                        and not beam["stack"].is_empty
                        and beam["current_rxn_obj"].reactant_templates[pos].match(beam["stack"].top())
                    )
                    if can_pop:
                        source_mask[0] = True

                    if not source_mask.any():
                        continue

                    source_logits[~source_mask] = -float("inf")
                    log_probs = F.log_softmax(source_logits, dim=-1)
                    topk = min(beam_width, int(source_mask.sum().item()))
                    top_vals, top_idxs = log_probs.topk(topk)

                    for val, idx in zip(top_vals, top_idxs):
                        source_id = idx.item()
                        nb = clone_beam(beam)
                        nb["log_prob"] += val.item()

                        if source_id == 0:  # Pop
                            pop_mol, _ = nb["stack"].pop()
                            nb["collected_reactants"][pos] = pop_mol
                            nb["token_types"].append(TokenType.POP.value)
                            nb["token_values"].append(0)
                            nb["route"].tokens.append(RouteToken(TokenType.POP))
                        else:  # BB
                            bb_idx = source_id - 1
                            nb["collected_reactants"][pos] = bb_mols[bb_idx]
                            nb["token_types"].append(TokenType.BB.value)
                            nb["token_values"].append(bb_idx)
                            nb["route"].tokens.append(RouteToken(TokenType.BB, bb_idx))

                        nb["remaining_positions"] = nb["remaining_positions"][1:]
                        nb["expect"] = "source" if nb["remaining_positions"] else "execute"
                        next_beams.append(nb)

            # Execute reactions for beams that are ready (deterministic, no branching)
            post_exec = []
            for nb in next_beams:
                if nb["expect"] == "execute":
                    if try_execute(nb):
                        post_exec.append(nb)
                    # else: dead beam, discard
                else:
                    post_exec.append(nb)

            # Prune to beam_width by log probability
            post_exec.sort(key=lambda b: b["log_prob"], reverse=True)
            active_beams = post_exec[:beam_width]

        # No auto-append END at max_steps. A beam that never emitted a valid
        # END token is dropped (decode failure). This is consistent with the
        # greedy decoder and matches the "model must predict END" semantics.

        # Return best completed route
        valid = [b for b in completed if b["route"].final_product is not None]
        if not valid:
            return None
        valid.sort(key=lambda b: b["log_prob"], reverse=True)
        return valid[0]["route"]

    @torch.no_grad()
    def decode_z_to_smiles(
        self,
        z: torch.Tensor,          # (B, latent_dim)
        compat_matrix,
        bb_mols: list,
        temperature: float = 1.0,
        greedy: bool = False,
        beam_width: int = 1,
    ) -> list[Optional[str]]:
        """Decode a batch of z vectors to SMILES strings.

        Args:
            beam_width: if > 1, use beam search instead of greedy/sampling.

        Decodes one at a time (autoregressive is sequential).
        Returns list of SMILES (or None for failed decodings).
        """
        B = z.shape[0]
        results = []
        for i in range(B):
            z_i = z[i:i+1]  # (1, latent_dim)
            if beam_width > 1:
                route = self.decode_beam_search(
                    z_i, compat_matrix, bb_mols,
                    beam_width=beam_width,
                )
            else:
                route = self.decode_autoregressive(
                    z_i, compat_matrix, bb_mols,
                    temperature=temperature, greedy=greedy,
                )
            if route is not None and route.final_product is not None:
                results.append(route.final_product)
            else:
                results.append(None)
        return results
