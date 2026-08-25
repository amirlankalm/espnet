# Evidence for `MiniOmniE2EModel.forward` audio assembly

This branch is not part of any pull request. It only hosts artifacts so they can be
linked from a PR discussion.

All three audio files are what `MiniOmniE2EModel.forward()` returns for the **same**
input — the first 2.00 s of `test_utils/ctc_align_test.wav` — decoded from the MP3 bytes
the method hands back.

| file | what it is | duration |
|---|---|---|
| `mini-omni-BEFORE-master.wav` | `master`: per-chunk SNAC decode, per-chunk MP3 files concatenated | 12.194 s |
| `mini-omni-AFTER-fix.wav` | with the fix: one decode, one encode | 10.496 s |
| `mini-omni-REFERENCE-full-sequence-decode.wav` | raw PCM from a single full-sequence `snacmodel.decode()`, never passed through MP3 | 10.496 s |

`before_after.png` plots the first two, shading every run quieter than
`|sample| <= 8` lasting at least 8 ms.

The "after" file and the reference are the same length and correlate at 0.998957; the
"before" file correlates at 0.041 with the reference because the silence inserted at each
chunk boundary progressively shifts everything after it.
