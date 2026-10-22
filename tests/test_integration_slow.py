"""Integración real con un modelo pequeño; se ejecuta solo con `-m slow`."""

import pytest
import torch

import server


@pytest.mark.slow
def test_gpt2_forward_con_kv_cache_real():
    state = server.load_model("gpt2")
    ids = server.build_input_ids(state["tokenizer"], "La capital de Francia es", chat=False)

    logits, activations, attention, decode_state, past = server.forward_once(
        ids,
        state,
        None,
        context_ids=ids[0].tolist(),
        use_cache=True,
    )

    assert state["key"] == "gpt2"
    assert logits.shape[-2] == ids.shape[-1]
    assert activations
    assert len(activations) == len(state["labels"])
    assert attention["layers"]
    assert len(attention["tokens"]) == ids.shape[-1]
    assert past is not None

    next_token = int(torch.argmax(logits[0, -1]).item())
    context = [*ids[0].tolist(), next_token]
    logits2, activations2, attention2, _, past2 = server.forward_once(
        torch.tensor([[next_token]]),
        state,
        decode_state,
        context_ids=context,
        past=past,
        use_cache=True,
    )

    assert logits2.shape[-2] == 1
    assert activations2
    assert len(attention2["tokens"]) == len(context)
    assert past2 is not None
