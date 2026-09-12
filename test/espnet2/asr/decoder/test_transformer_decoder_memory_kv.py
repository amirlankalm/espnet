import pytest
import torch

from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.legacy.nets.batch_beam_search import BatchBeamSearch
from espnet2.legacy.nets.beam_search import BeamSearch
from espnet2.legacy.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,
)
from espnet2.legacy.nets.pytorch_backend.transformer.mask import subsequent_mask


def _decoder(**kwargs):
    torch.manual_seed(0)
    decoder = TransformerDecoder(
        vocab_size=10,
        encoder_output_size=8,
        attention_heads=2,
        linear_units=12,
        num_blocks=2,
        dropout_rate=0.0,
        positional_dropout_rate=0.0,
        self_attention_dropout_rate=0.0,
        src_attention_dropout_rate=0.0,
        **kwargs,
    )
    return decoder.eval()


@pytest.mark.parametrize("use_sdpa", [False, True])
@pytest.mark.parametrize("qk_norm", [False, True])
def test_forward_with_kv_matches_forward(use_sdpa, qk_norm):
    torch.manual_seed(0)
    attn = MultiHeadedAttention(2, 8, 0.0, qk_norm=qk_norm, use_sdpa=use_sdpa)
    attn.eval()
    query = torch.randn(3, 1, 8)
    memory = torch.randn(3, 7, 8)
    mask = torch.ones(3, 1, 7, dtype=torch.bool)
    mask[1, :, 5:] = False

    expected = attn(query, memory, memory, mask)
    k, v = attn.forward_kv(memory, memory)
    torch.testing.assert_close(attn.forward_with_kv(query, k, v, mask), expected)

    # one memory shared by every query: transformed once, broadcast at use
    shared = memory[:1].expand(3, 7, 8)
    expected = attn(query, shared, shared, None)
    k, v = attn.forward_kv(memory[:1], memory[:1])
    assert k.size(0) == 1 and v.size(0) == 1
    torch.testing.assert_close(attn.forward_with_kv(query, k, v, None), expected)


def test_forward_one_step_with_memory_kv():
    decoder = _decoder()
    memory = torch.randn(2, 6, 8)
    memory_mask = torch.ones(2, 1, 6, dtype=torch.bool)
    memory_mask[1, :, 4:] = False
    ys = torch.randint(0, 10, (2, 3))
    ys_mask = subsequent_mask(3).unsqueeze(0)

    expected, cache = decoder.forward_one_step(ys, ys_mask, memory, memory_mask)
    kv = decoder.memory_kv(memory)
    assert len(kv) == 2
    assert kv[0][0].shape == (2, 2, 6, 4) and kv[0][1].shape == (2, 2, 6, 4)
    got, cache_kv = decoder.forward_one_step(
        ys, ys_mask, memory, memory_mask, memory_kv=kv
    )
    torch.testing.assert_close(got, expected)
    for a, b in zip(cache, cache_kv):
        torch.testing.assert_close(a, b)


def test_memory_kv_expanded_memory():
    decoder = _decoder()
    x = torch.randn(6, 8)
    xs = x.expand(5, 6, 8)  # one utterance over 5 hypotheses, stride 0
    kv = decoder.memory_kv(xs)
    assert kv[0][0].size(0) == 1
    ys = torch.randint(0, 10, (5, 2))
    ys_mask = subsequent_mask(2).unsqueeze(0)
    expected, _ = decoder.forward_one_step(ys, ys_mask, xs)
    got, _ = decoder.forward_one_step(ys, ys_mask, xs, memory_kv=kv)
    torch.testing.assert_close(got, expected)


def test_memory_kv_is_reused_and_invalidated():
    decoder = _decoder()
    x = torch.randn(6, 8)
    kv = decoder.memory_kv(x.unsqueeze(0))
    # a new view of the same storage, as score() builds at every step: reused
    assert decoder.memory_kv(x.unsqueeze(0)) is kv
    # another tensor: transformed again
    other = torch.randn(1, 6, 8)
    kv_other = decoder.memory_kv(other)
    assert kv_other is not kv
    # modified in place: transformed again
    other.add_(1.0)
    assert decoder.memory_kv(other) is not kv_other
    # the beam search resets the state before each utterance: dropped
    kv_other = decoder.memory_kv(other)
    decoder.init_state(other)
    assert decoder._memory_kv_cache is None
    assert decoder.memory_kv(other) is not kv_other
    # disabled
    decoder.cache_memory_kv = False
    assert decoder.memory_kv(other) is None


def test_batch_score_and_score_match_uncached():
    decoder = _decoder()
    xs = torch.randn(3, 6, 8)
    xs_mask = torch.ones(3, 1, 6, dtype=torch.bool)
    xs_mask[2, :, 4:] = False
    ys = torch.randint(0, 10, (3, 2))
    ys_mask = subsequent_mask(2).unsqueeze(0)

    logp, states = decoder.batch_score(ys, [None] * 3, xs, xs_mask=xs_mask)
    expected, _ = decoder.forward_one_step(ys, ys_mask, xs, memory_mask=xs_mask)
    torch.testing.assert_close(logp, expected)
    kv = decoder._memory_kv_cache[2]

    # the next step reuses the transformed key and value
    ys = torch.cat([ys, logp.argmax(-1, keepdim=True)], dim=1)
    decoder.batch_score(ys, states, xs, xs_mask=xs_mask)
    assert decoder._memory_kv_cache[2] is kv

    x = xs[0]
    ys = torch.randint(0, 10, (2,))
    logp, _ = decoder.score(ys, None, x)
    expected, _ = decoder.forward_one_step(
        ys.unsqueeze(0), ys_mask, x.unsqueeze(0), memory_kv=None
    )
    torch.testing.assert_close(logp, expected.squeeze(0))


@pytest.mark.parametrize("search_class", [BeamSearch, BatchBeamSearch])
def test_beam_search_same_result_with_and_without_cache(search_class):
    decoder = _decoder()
    x = torch.randn(9, 8)
    results = []
    for cache in (True, False):
        decoder.cache_memory_kv = cache
        search = search_class(
            scorers={"decoder": decoder},
            weights={"decoder": 1.0},
            beam_size=3,
            vocab_size=10,
            sos=9,
            eos=9,
            token_list=[str(i) for i in range(10)],
            pre_beam_score_key=None,
        )
        with torch.no_grad():
            results.append(search(x, maxlenratio=0.5))
    with_cache, without_cache = results
    assert [h.yseq.tolist() for h in with_cache] == [
        h.yseq.tolist() for h in without_cache
    ]
    torch.testing.assert_close(
        torch.stack([h.score for h in with_cache]),
        torch.stack([h.score for h in without_cache]),
    )
