"""Compare MFControl's threshold and clustering with PyCBC's.

Both sides see the same correlation series: matchedfilter's full output
(CorrelationFilter) is fed to PyCBC's ThresholdCluster, while MFControl uses
matchedfilter's peak kernel on the same spectra. Any difference is then in
binning, thresholding or the neighbour comparisons, not in FFT rounding.
"""

import numpy

import matchedfilter as mf
from pycbc.events import ThresholdCluster
from pycbc.types import Array

from mf_inspiral.control import MFControl


class FakeSegment(object):
    def __init__(self, data, analyze):
        self._data = data
        self.analyze = analyze

    def numpy(self):
        return self._data


ANALYZE_SLICES = [slice(4096, 61440), slice(1000, 50001)]


def test_matches_pycbc_threshold_cluster():
    for seed in range(5):
        for analyze in ANALYZE_SLICES:
            check_against_pycbc(seed, analyze)


def check_against_pycbc(seed, analyze):
    n, window = 1 << 16, 256
    rng = numpy.random.default_rng(seed)
    data = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) \
        .astype(numpy.complex64)
    tmpl = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) \
        .astype(numpy.complex64)

    # kmin = 1 with no low-frequency cutoff: zero DC so both see the same data
    data[0] = 0
    sigmasq = 1.0
    delta_f = 1.0 / n
    norm = 4.0 * delta_f / numpy.sqrt(sigmasq)
    # Threshold at roughly the median bin maximum, so many bins are dismissed
    threshold = 1.0 * numpy.sqrt(n) * numpy.sqrt(2 * numpy.log(window)) * norm

    control = MFControl([FakeSegment(data, analyze)], Array(tmpl), None,
                        threshold, window, n, delta_f)
    control.set_template()
    _, idx, snrv = control.matched_filter_and_cluster(0, sigmasq)

    full = mf.CorrelationFilter(n, device="cpu")
    full.set_data(numpy.where(numpy.arange(n) < (n + 1) // 2, data, 0)[None])
    full.set_templates(tmpl[None])
    z = full.run()[0, 0].copy()
    tc = ThresholdCluster(Array(z[analyze]))
    ref_v, ref_i = tc.threshold_and_cluster(threshold / norm, window)

    assert 0 < len(ref_i) < (analyze.stop - analyze.start) // window
    numpy.testing.assert_array_equal(idx, ref_i.astype(numpy.int64))
    numpy.testing.assert_array_equal(snrv, ref_v)


if __name__ == '__main__':
    test_matches_pycbc_threshold_cluster()
    print("test_matches_pycbc_threshold_cluster: OK")
