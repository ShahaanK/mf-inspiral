# mf-inspiral

A pycbc_inspiral-compatible executable that computes SNR-only triggers, using
the [matchedfilter](https://github.com/ahnitz/matchedfilter) package for the
correlation, threshold and per-segment clustering. Everything else (data
conditioning, PSDs, templates, template norms, PSD variation, output) is
PyCBC's.

Status: CPU path only. Chi-squared and the other vetoes are not implemented
and their options are refused.

```
bin/mf_inspiral [pycbc_inspiral options] --mf-device cpu
```

Requires PyCBC 2.10.0 and matchedfilter 0.1.0a9.

Tests (no pytest needed):

```
PYTHONPATH=. python tests/test_cluster.py
```

Compare an output with a pycbc_inspiral reference:

```
python tools/compare_triggers.py REFERENCE.hdf NEW.hdf
```

Licence: not yet decided.
