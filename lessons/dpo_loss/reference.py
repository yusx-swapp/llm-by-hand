    def _compute_loss(self, model, inputs, return_outputs):
        mode = "train" if self.model.training else "eval"
        device = self.accelerator.device

        _non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}
        model_kwargs = {k: v for k, v in inputs.items() if k not in _non_model_keys}
        model_kwargs["use_cache"] = False
        outputs = model(**model_kwargs)

        input_ids = inputs["input_ids"]
        completion_mask = inputs["completion_mask"]
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_completion_mask = completion_mask[..., 1:].contiguous()
        per_token_logps = selective_log_softmax(shift_logits, shift_labels)
        per_token_logps[shift_completion_mask == 0] = 0.0  # mask out non-completion tokens
        if self.ld_alpha is None:
            logps = per_token_logps.sum(dim=1)  # sum over sequence length
        else:
            comp_pos = shift_completion_mask.cumsum(dim=1)
            comp_lens = shift_completion_mask.sum(dim=1).long()
            chosen_lens, rejected_lens = comp_lens.chunk(2, dim=0)
            shared_lens = torch.minimum(chosen_lens, rejected_lens)
            shared_lens = torch.cat([shared_lens, shared_lens], dim=0).to(device)
            shared_mask = (comp_pos > 0) & (comp_pos <= shared_lens.unsqueeze(1))  # shared: 1 <= pos <= shared_len
            tail_mask = comp_pos > shared_lens.unsqueeze(1)  # tail: pos > shared_len
            shared_logps = (per_token_logps * shared_mask).sum(dim=1)
            tail_logps = (per_token_logps * tail_mask).sum(dim=1)
            logps = shared_logps + self.ld_alpha * tail_logps
        chosen_logps, rejected_logps = logps.chunk(2, dim=0)  # batch is [chosen, rejected]

        if self.precompute_ref_logps:
            ref_chosen_logps, ref_rejected_logps = inputs["ref_chosen_logps"], inputs["ref_rejected_logps"]
        else:
            # When gradient checkpointing is enabled with use_reentrant=True (default), calling the model inside a
            # torch.no_grad() block triggers a harmless PyTorch warning ("None of the inputs have requires_grad=True").
            # Temporarily disable checkpointing to avoid this warning during inference.
            with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
                if is_peft_model(model) and self.ref_model is None:
                    # When training a PEFT adapter, how we obtain the reference depends on the setup:
                    # - New adapter: disabling adapters yields the base model.
                    # - Re-training an existing adapter: an initial copy is loaded under the name "ref".
                    model = self.accelerator.unwrap_model(model)
                    with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
                        ref_outputs = self.model(**model_kwargs)
                else:
                    ref_outputs = self.ref_model(**model_kwargs)

            ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()
            ref_per_token_logps = selective_log_softmax(ref_shift_logits, shift_labels)
            ref_per_token_logps[shift_completion_mask == 0] = 0.0  # mask out non-completion tokens
            if self.ld_alpha is None:
                ref_logps = ref_per_token_logps.sum(dim=1)  # sum over sequence length
            else:
                # reuse comp_pos/shared_mask/tail_mask computed above (they depend only on completion_mask)
                ref_shared_logps = (ref_per_token_logps * shared_mask).sum(dim=1)
                ref_tail_logps = (ref_per_token_logps * tail_mask).sum(dim=1)
                ref_logps = ref_shared_logps + self.ld_alpha * ref_tail_logps
            ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)  # batch is [chosen, rejected]

        # Get the log ratios for the chosen and rejected responses
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps

        if self.f_divergence_type == "reverse_kl":  # standard DPO
            chosen_scores = chosen_logratios
            rejected_scores = rejected_logratios
        elif self.f_divergence_type == "forward_kl":
            # f'(t) = 1 - 1/t  -> drop constant -> -exp(-logratio)
            chosen_scores = -torch.exp(-chosen_logratios)
            rejected_scores = -torch.exp(-rejected_logratios)
        elif self.f_divergence_type == "js_divergence":
            # f'(t) = log(2t/(t+1)) -> drop log 2
            chosen_scores = F.logsigmoid(chosen_logratios)
            rejected_scores = F.logsigmoid(rejected_logratios)
        elif self.f_divergence_type == "alpha_divergence":
            # alpha-divergence: f'(t) = (t^(α-1) - 1)/(α-1)
            if abs(self.f_alpha_divergence_coef - 1.0) < 1e-6:  # limit case f'(t) -> log(t), fall back to reverse_kl
                chosen_scores = chosen_logratios
                rejected_scores = rejected_logratios
            else:
                coef = 1.0 / (self.f_alpha_divergence_coef - 1.0)
                t_chosen = (self.f_alpha_divergence_coef - 1.0) * chosen_logratios
                t_rejected = (self.f_alpha_divergence_coef - 1.0) * rejected_logratios
                dtype = t_chosen.dtype
                # Clamp max so exp(.) stays representable after casting back
                clamp_max = {torch.float16: 11.0, torch.bfloat16: 80.0, torch.float32: 80.0}[dtype]
                t_chosen_float = torch.clamp(t_chosen.float(), max=clamp_max)
                t_rejected_float = torch.clamp(t_rejected.float(), max=clamp_max)
                chosen_scores = torch.exp(t_chosen_float).to(dtype) * coef
                rejected_scores = torch.exp(t_rejected_float).to(dtype) * coef
        else:
            raise ValueError(f"Unknown f_divergence_type: {self.f_divergence_type}")

        delta_score = chosen_scores - rejected_scores

        loss = 0.0
        for loss_type, loss_weight in zip(self.loss_types, self.loss_weights, strict=True):
            if loss_type == "sigmoid":
                per_sequence_loss = -F.logsigmoid(self.beta * delta_score)

            elif loss_type == "hinge":
                per_sequence_loss = torch.relu(1 - self.beta * delta_score)

            elif loss_type == "ipo":
                # IPO uses sequence-level log-prob differences; in code these are token-summed over the completion,
                # which makes the squared loss scale with completion length. We therefore normalize by the number of
                # completion tokens (average per token) to make β/loss comparable across variable lengths. This length
                # normalization is not explicitly discussed in the IPO paper; we confirmed this choice with the IPO
                # authors, and the results reported in the paper correspond to this normalized form.
                chosen_mask, rejected_mask = completion_mask.chunk(2, dim=0)
                chosen_avg_score = chosen_scores / chosen_mask.sum(dim=1).clamp(min=1.0)
                rejected_avg_score = rejected_scores / rejected_mask.sum(dim=1).clamp(min=1.0)
                ipo_delta = chosen_avg_score - rejected_avg_score
                # (Eq. 17) of the paper where beta is the regularization parameter for the IPO loss, denoted by τ.
                per_sequence_loss = (ipo_delta - 1 / (2 * self.beta)) ** 2

            elif loss_type == "exo_pair":
                # Implements EXO-pref from the paper https://huggingface.co/papers/2402.00856, (Eq. 16)
                # Minimize KL(p_fθ || p_rh) for K=2; p_fθ = softmax(βπ * (log πθ − log π_ref)) over {chosen, rejected}
                # p_rh = [(1−ε), ε]; expanded KL gives the weighted logsigmoid form below
                epsilon = torch.tensor(self.label_smoothing, device=device)
                qw = torch.sigmoid(self.beta * delta_score)
                log_qw = F.logsigmoid(self.beta * delta_score)
                log_pw = torch.log1p(-epsilon)
                ql = torch.sigmoid(-self.beta * delta_score)
                log_ql = F.logsigmoid(-self.beta * delta_score)
                log_pl = torch.log(epsilon)
                per_sequence_loss = qw * (log_qw - log_pw) + ql * (log_ql - log_pl)

            elif loss_type == "nca_pair":
                chosen_rewards = self.beta * chosen_scores
                rejected_rewards = self.beta * rejected_scores
                per_sequence_loss = (
                    -F.logsigmoid(chosen_rewards)
                    - 0.5 * F.logsigmoid(-chosen_rewards)
                    - 0.5 * F.logsigmoid(-rejected_rewards)
                )

            elif loss_type == "robust":
                clean_loss_term = -(1 - self.label_smoothing) * F.logsigmoid(self.beta * delta_score)
                flipped_loss_term = -self.label_smoothing * F.logsigmoid(-self.beta * delta_score)
                per_sequence_loss = (clean_loss_term - flipped_loss_term) / (1 - 2 * self.label_smoothing)

            elif loss_type == "bco_pair":
                chosen_rewards = self.beta * chosen_scores
                rejected_rewards = self.beta * rejected_scores
                per_sequence_loss = -F.logsigmoid(chosen_rewards) - F.logsigmoid(-rejected_rewards)

            elif loss_type == "sppo_hard":
                # In the paper (https://huggingface.co/papers/2405.00675), SPPO employs a soft probability approach,
                # estimated using the PairRM score. The probability calculation is conducted outside of the trainer
                # class. The version described here is the hard probability version, where P in Equation (4.7) of
                # Algorithm 1 is set to 1 for the winner and 0 for the loser.
                winner_margin_error = (chosen_scores - 0.5 / self.beta) ** 2
                loser_margin_error = (rejected_scores + 0.5 / self.beta) ** 2
                per_sequence_loss = winner_margin_error + loser_margin_error

            elif loss_type == "aot":
                logratios = chosen_logps - rejected_logps
                ref_logratios = ref_chosen_logps - ref_rejected_logps
                logratios_sorted, _ = torch.sort(logratios, dim=0)
                ref_logratios_sorted, _ = torch.sort(ref_logratios, dim=0)
                delta = logratios_sorted - ref_logratios_sorted
                per_sequence_loss = (
                    -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * delta) * self.label_smoothing
                )

            elif loss_type == "aot_unpaired":
                chosen_logratios_sorted, _ = torch.sort(chosen_logratios, dim=0)
                rejected_logratios_sorted, _ = torch.sort(rejected_logratios, dim=0)
                delta = chosen_logratios_sorted - rejected_logratios_sorted
                per_sequence_loss = (
                    -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * delta) * self.label_smoothing
                )

            elif loss_type == "apo_zero":
                # Eqn (7) of the APO paper (https://huggingface.co/papers/2408.06266)
                # Use this loss when you believe the chosen outputs are better than your model's default output
                # Increase chosen likelihood and decrease rejected likelihood
                losses_chosen = 1 - torch.sigmoid(self.beta * chosen_logratios)
                losses_rejected = torch.sigmoid(self.beta * rejected_logratios)
                per_sequence_loss = losses_chosen + losses_rejected

            elif loss_type == "apo_down":
                # Eqn (8) of the APO paper (https://huggingface.co/papers/2408.06266)
                # Use this loss when you believe the chosen outputs are worse than your model's default output.
                # Decrease chosen likelihood and decrease rejected likelihood more
                losses_chosen = torch.sigmoid(self.beta * chosen_logratios)
                losses_rejected = 1 - torch.sigmoid(self.beta * delta_score)
                per_sequence_loss = losses_chosen + losses_rejected

            elif loss_type == "discopop":
                # Eqn (5) of the DiscoPOP paper (https://huggingface.co/papers/2406.08414)
                logits = delta_score * self.beta
                # Modulate the mixing coefficient based on the log ratio magnitudes
                log_ratio_modulation = torch.sigmoid(logits / self.args.discopop_tau)
                logistic_component = -F.logsigmoid(logits)
                exp_component = torch.exp(-logits)
                # Blend between logistic and exponential component based on log ratio modulation
                per_sequence_loss = (
                    logistic_component * (1 - log_ratio_modulation) + exp_component * log_ratio_modulation
                )

            elif loss_type == "sft":
                chosen_logits, _ = shift_logits.chunk(2, dim=0)
                chosen_labels, _ = shift_labels.chunk(2, dim=0)
                chosen_mask, _ = shift_completion_mask.chunk(2, dim=0)
                batch_loss = F.cross_entropy(chosen_logits[chosen_mask.bool()], chosen_labels[chosen_mask.bool()])
                # Implementation convenience: expand the scalar SFT loss to a per-sequence tensor so it matches the
                # shape of other losses; only the mean is used, so this is a no-op numerically.
                per_sequence_loss = batch_loss.expand(chosen_logits.size(0))

            else:
                raise ValueError(
                    f"Unknown loss type: {loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'exo_pair', "
                    "'nca_pair', 'robust', 'bco_pair', 'sppo_hard', 'aot', 'aot_unpaired', 'apo_zero', 'apo_down', "
                    "'discopop', 'sft']"
                )

            if self.use_weighting:
                # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
                completion_lengths = shift_completion_mask.sum(dim=1).clamp_min(1)
                with torch.no_grad():
                    lse1 = torch.logsumexp(shift_logits, dim=-1)
                    lse2 = torch.logsumexp(2.0 * shift_logits, dim=-1)
                    log_denom = lse2 - 2.0 * lse1
                    aligned_logps = (per_token_logps - log_denom) * shift_completion_mask
                mean_logps = aligned_logps.sum(dim=1) / completion_lengths
                weights = torch.exp(mean_logps)
                chosen_weights, rejected_weights = weights.chunk(2, dim=0)
                per_sequence_loss *= chosen_weights * rejected_weights

            loss += per_sequence_loss.mean() * loss_weight

        # Log the metrics
        # Entropy
        per_token_entropy = entropy_from_logits(shift_logits.detach())
        entropy = per_token_entropy[shift_completion_mask.bool()].mean()
        entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
        self._metrics[mode]["entropy"].append(entropy)

        # Number of tokens
        if mode == "train":
            num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Average logits for chosen and rejected completions
        chosen_logits, rejected_logits = shift_logits.detach().chunk(2, dim=0)
        chosen_mask, rejected_mask = shift_completion_mask.chunk(2, dim=0)
        total_chosen_logits = chosen_logits[chosen_mask.bool()].mean(-1).sum()
        total_chosen_tokens = chosen_mask.sum()
        total_rejected_logits = rejected_logits[rejected_mask.bool()].mean(-1).sum()
        total_rejected_tokens = rejected_mask.sum()
        total_chosen_logits = self.accelerator.gather_for_metrics(total_chosen_logits).sum().item()
        total_chosen_tokens = self.accelerator.gather_for_metrics(total_chosen_tokens).sum().item()
        total_rejected_logits = self.accelerator.gather_for_metrics(total_rejected_logits).sum().item()
        total_rejected_tokens = self.accelerator.gather_for_metrics(total_rejected_tokens).sum().item()
        avg_chosen_logits = total_chosen_logits / total_chosen_tokens if total_chosen_tokens > 0 else 0.0
        avg_rejected_logits = total_rejected_logits / total_rejected_tokens if total_rejected_tokens > 0 else 0.0
        self._metrics[mode]["logits/chosen"].append(avg_chosen_logits)
        self._metrics[mode]["logits/rejected"].append(avg_rejected_logits)

        # Token accuracy for the chosen completions
        predictions = chosen_logits.argmax(dim=-1)
        chosen_mask = shift_completion_mask[: len(shift_completion_mask) // 2].bool()
        chosen_labels = shift_labels[: len(shift_labels) // 2]
        correct_predictions = (predictions == chosen_labels) & chosen_mask
        total_tokens = chosen_mask.sum()
        correct_tokens = correct_predictions.sum()
        correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
        total_tokens = self.accelerator.gather_for_metrics(total_tokens)
        total_sum = total_tokens.sum()
        accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
        self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        # Rewards for chosen and rejected completions
        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()
        agg_chosen_rewards = self.accelerator.gather(chosen_rewards)
        agg_rejected_rewards = self.accelerator.gather(rejected_rewards)
        self._metrics[mode]["rewards/chosen"].append(agg_chosen_rewards.mean().item())
        self._metrics[mode]["rewards/rejected"].append(agg_rejected_rewards.mean().item())

        # Reward accuracy
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        agg_reward_accuracies = self.accelerator.gather(reward_accuracies)
        self._metrics[mode]["rewards/accuracies"].append(agg_reward_accuracies.mean().item())

        # Reward margins
        margins = chosen_rewards - rejected_rewards
        agg_margins = self.accelerator.gather(margins)
        self._metrics[mode]["rewards/margins"].append(agg_margins.mean().item())

        # Average log probabilities for chosen and rejected completions
        self._metrics[mode]["logps/chosen"].append(self.accelerator.gather(chosen_logps).mean().item())
        self._metrics[mode]["logps/rejected"].append(self.accelerator.gather(rejected_logps).mean().item())

        return (loss, outputs) if return_outputs else loss
