    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids = [example["input_ids"] for example in examples]
        batch_seq_lengths = [example["seq_lengths"] for example in examples] if "seq_lengths" in examples[0] else None
        labels = [example.get("labels", example["input_ids"]) for example in examples]
        completion_mask = (
            [example["completion_mask"] for example in examples]
            if self.completion_only_loss and "completion_mask" in examples[0]
            else None
        )
        assistant_masks = (
            [example["assistant_masks"] for example in examples] if "assistant_masks" in examples[0] else None
        )

        # Truncate per sequence if necessary
        if self.max_length is not None and not self.padding_free:
            if self.truncation_mode == "keep_start":
                sl = slice(None, self.max_length)
            elif self.truncation_mode == "keep_end":
                sl = slice(-self.max_length, None)
            else:
                raise ValueError(
                    f"Unsupported truncation mode: {self.truncation_mode}, expected 'keep_start' or 'keep_end'"
                )
            input_ids = [ids[sl] for ids in input_ids]
            labels = [lbl[sl] for lbl in labels]
            if completion_mask is not None:
                completion_mask = [m[sl] for m in completion_mask]
            if assistant_masks is not None:
                assistant_masks = [m[sl] for m in assistant_masks]

        # Convert to tensor
        input_ids = [torch.tensor(ids) for ids in input_ids]
        labels = [torch.tensor(lbl) for lbl in labels]
        if completion_mask is not None:
            completion_mask = [torch.tensor(m) for m in completion_mask]
        if assistant_masks is not None:
            assistant_masks = [torch.tensor(m) for m in assistant_masks]

        # For padding-free, we should NOT create attention_mask as it causes FlashAttention to ignore position_ids and
        # compute wrong cu_seq_lens from the all-1s mask
        if self.padding_free:
            if batch_seq_lengths is not None:
                position_ids = self.get_position_ids_from_packed_seq_lengths(batch_seq_lengths)
            else:
                position_ids = [torch.arange(len(ids)) for ids in input_ids]
        else:
            attention_mask = [torch.ones_like(ids) for ids in input_ids]

        # If padding_free, flatten everything into a single sequence
        output = {}
        if self.padding_free:
            input_ids = [torch.cat(input_ids, dim=0)]
            labels = [torch.cat(labels, dim=0)]
            position_ids = [torch.cat(position_ids, dim=0)]
            if completion_mask is not None:
                completion_mask = [torch.cat(completion_mask, dim=0)]
            if assistant_masks is not None:
                assistant_masks = [torch.cat(assistant_masks, dim=0)]

        # Pad
        output["input_ids"] = pad(
            input_ids,
            padding_value=self.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        output["labels"] = pad(
            labels, padding_value=-100, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
        )
        if self.padding_free:
            output["position_ids"] = pad(
                position_ids, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
            output["labels"][output["position_ids"] == 0] = -100
        else:
            output["attention_mask"] = pad(
                attention_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
        if completion_mask is not None:
            completion_mask = pad(
                completion_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
            output["labels"][completion_mask == 0] = -100  # mask everything that is not in the completion
        if assistant_masks is not None:
            assistant_masks = pad(
                assistant_masks, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
            output["labels"][assistant_masks == 0] = -100
        return output
