"""``early_stop=k`` must leave the k-best list unchanged and take fewer steps."""

from test.espnet2.legacy.test_beam_search import prepare, transformer_args

import pytest
import torch

from espnet2.legacy.nets.batch_beam_search import BatchBeamSearch
from espnet2.legacy.nets.beam_search import BeamSearch
from espnet2.legacy.nets.scorers.ctc import CTCPrefixScorer
from espnet2.legacy.nets.scorers.length_bonus import LengthBonus


def _setup(ctc_weight, bonus=0.0):
    torch.manual_seed(123)
    model, x, ilens, y, data, train_args = prepare(
        transformer_args, mtlalpha=ctc_weight
    )
    model.eval()
    token_list = train_args.token_list
    scorers = {"decoder": model.decoder, "length_bonus": LengthBonus(len(token_list))}
    weights = {"decoder": 1.0 - ctc_weight, "length_bonus": bonus}
    if ctc_weight > 0:
        scorers["ctc"] = CTCPrefixScorer(ctc=model.ctc, eos=model.eos)
        weights["ctc"] = ctc_weight
    with torch.no_grad():
        enc, enc_lens = model.encode(x, torch.tensor(ilens))
    encs = [enc[b, : enc_lens[b]] for b in range(x.size(0))]
    common = dict(
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        token_list=token_list,
        sos=model.sos,
        eos=model.eos,
        beam_size=3,
        pre_beam_score_key=None if ctc_weight == 1.0 else "full",
    )
    return encs, common


def _count_steps(beam_search):
    """Wrap ``search`` so that the number of decoding steps can be read back."""
    calls = []
    original = beam_search.search

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    beam_search.search = counted
    return calls


def _same_top(expected, actual, k):
    """The k best hypotheses must be identical; beyond k the lists may differ."""
    assert len(actual) >= min(k, len(expected))
    for e, a in zip(expected[:k], actual[:k]):
        assert e.yseq.tolist() == a.yseq.tolist()
        assert float(e.score) == pytest.approx(float(a.score), rel=1e-5)


# NOTE: a constant maxlen longer than the encoder output is only valid without
# CTC, whose prefix scorer indexes time by hypothesis length; and at a hard cap
# the search force-appends <eos> to hypotheses that never paid its
# log-probability, which is the one place a capped run can differ from an
# early-stopped one that never reaches the cap. -30 is beyond any natural end.
@pytest.mark.parametrize(
    "ctc_weight, maxlenratio", [(0.0, 1.0), (0.5, 1.0), (0.0, -30)]
)
@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("cls", [BeamSearch, BatchBeamSearch])
def test_early_stop_same_kbest_fewer_steps(cls, ctc_weight, k, maxlenratio):
    encs, common = _setup(ctc_weight)
    full = cls(**common, early_stop=0).eval()
    early = cls(**common, early_stop=k).eval()
    steps_full, steps_early = _count_steps(full), _count_steps(early)
    with torch.no_grad():
        for enc in encs:
            expected = full(x=enc, maxlenratio=maxlenratio, minlenratio=0.0)
            actual = early(x=enc, maxlenratio=maxlenratio, minlenratio=0.0)
            _same_top(expected, actual, k)
    assert len(steps_early) <= len(steps_full)


@pytest.mark.parametrize("ctc_weight", [0.0, 0.5])
@pytest.mark.parametrize("k", [1, 3])
def test_early_stop_batched_utterances(ctc_weight, k):
    """A batch of utterances with early_stop matches each utterance decoded alone."""
    encs, common = _setup(ctc_weight)
    reference = BatchBeamSearch(**common, early_stop=0).eval()
    batched = BatchBeamSearch(**common, early_stop=k).eval()
    lengths = torch.tensor([e.size(0) for e in encs])
    padded = torch.zeros(len(encs), int(lengths.max()), encs[0].size(1))
    for b, e in enumerate(encs):
        padded[b, : e.size(0)] = e
    with torch.no_grad():
        expected = [reference(x=e, maxlenratio=1.0, minlenratio=0.0) for e in encs]
        actual = batched(x=padded, x_lengths=lengths, maxlenratio=1.0, minlenratio=0.0)
    for exp, act in zip(expected, actual):
        _same_top(exp, act, k)


def test_early_stop_warns_when_not_exact(caplog):
    encs, common = _setup(0.0, bonus=0.5)
    with caplog.at_level("WARNING"):
        BeamSearch(**common, early_stop=1)
    assert "early_stop" in caplog.text
    caplog.clear()
    exact = {**common, "weights": {**common["weights"], "length_bonus": 0.0}}
    with caplog.at_level("WARNING"):
        BeamSearch(**exact, early_stop=1)
    assert "early_stop" not in caplog.text
