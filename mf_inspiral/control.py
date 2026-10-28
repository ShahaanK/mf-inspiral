"""Matched filtering of PyCBC data segments with the matchedfilter package.

The data and template spectra are the ones PyCBC prepares for its own
matched filter: overwhitened segments (s/S) and FilterBank templates, both
in PyCBC's DYN_RANGE_FAC units. matchedfilter returns the unnormalised
inverse FFT of data * conj(template), which is the same quantity PyCBC's
correlator and IFFT produce, so the PyCBC normalisation 4 df / sqrt(sigmasq)
converts it to SNR.
"""

from math import sqrt

import numpy

import matchedfilter as mf
from pycbc.filter.matchedfilter import get_cutoff_indices


def symmetric_cluster(index, value, nbins):
    """Apply PyCBC's symmetric window clustering to one row of bin maxima.

    ``index`` and ``value`` are matchedfilter's per-bin peak and its lag, with
    bins at or below threshold dismissed (index -1, value 0). The comparison
    rules follow PyCBC's parallel_thresh_cluster: an interior bin survives if
    it is louder than the bin before and at least as loud as the bin after;
    the first and last bins compare against their single neighbour. A
    dismissed neighbour is below threshold, so comparing against its zero
    gives the same answer as comparing against its true maximum.
    """
    above = index >= 0
    if nbins == 1:
        return above
    mag = value.real * value.real + value.imag * value.imag
    keep = above.copy()
    keep[0] &= mag[0] > mag[1]
    keep[-1] &= mag[-1] > mag[-2]
    if nbins > 2:
        keep[1:-1] &= (mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:])
    return keep


class MFControl(object):
    """Threshold and cluster each (template, segment) pair with matchedfilter.

    Parameters
    ----------
    segments : list of FrequencySeries
        Overwhitened data segments, as returned by fourier_segments() after
        division by their PSDs. Each must carry an ``analyze`` slice.
    template_mem : complex64 Array
        Length-tlen memory that FilterBank writes each template into.
    low_frequency_cutoff : float
        Lower edge of the correlation, in Hz.
    snr_threshold : float
        SNR above which a clustered peak is kept.
    cluster_window : int
        Clustering window, in samples.
    tlen, delta_f : int, float
        Time-domain length and frequency step of the segments.
    device : str
        matchedfilter device. Only the CPU path is supported so far.
    """

    def __init__(self, segments, template_mem, low_frequency_cutoff,
                 snr_threshold, cluster_window, tlen, delta_f, device='cpu'):
        if cluster_window < 1:
            raise ValueError("A positive cluster window is required")
        self.tlen = int(tlen)
        self.delta_f = delta_f
        self.snr_threshold = snr_threshold
        self.cluster_window = int(cluster_window)
        self.htilde = template_mem
        self.windows = [(s.analyze.start, s.analyze.stop) for s in segments]

        # PyCBC correlates only bins [kmin, kmax). Zeroing the data outside
        # that range makes the full-length product identical.
        self.kmin, self.kmax = get_cutoff_indices(low_frequency_cutoff, None,
                                                  delta_f, self.tlen)
        data = numpy.zeros((len(segments), self.tlen), dtype=numpy.complex64)
        for i, seg in enumerate(segments):
            data[i, self.kmin:self.kmax] = seg.numpy()[self.kmin:self.kmax]

        self.filter = mf.MatchedFilter(self.tlen, ndata=len(segments),
                                       ntemplates=1, device=device)
        self.filter.set_data(data)
        self._template_set = False

    def set_template(self):
        """Load the template currently held in template memory."""
        self.filter.set_templates(self.htilde.numpy()[None, :])
        self._template_set = True

    def matched_filter_and_cluster(self, segnum, template_norm):
        """Filter segment ``segnum`` against the loaded template.

        Returns ``(norm, idx, snrv)``: the SNR normalisation, the clustered
        peak indices relative to the start of the segment's analyze slice,
        and the unnormalised complex filter output at those peaks. Empty
        arrays mean nothing passed the threshold.
        """
        if not self._template_set:
            raise RuntimeError("set_template() must be called first")
        # math.sqrt, as PyCBC uses, keeps norm a Python float so that
        # snrv * norm stays complex64 under NumPy 2 promotion rules.
        norm = (4.0 * self.delta_f) / sqrt(template_norm)
        start, stop = self.windows[segnum]
        peaks = self.filter.run(binsize=self.cluster_window,
                                threshold=self.snr_threshold / norm,
                                window=(start, stop), data=(segnum, 1),
                                raw=True)
        index, value = peaks[0][0, 0], peaks[1][0, 0]
        keep = symmetric_cluster(index, value, len(index))
        idx = index[keep] - start
        snrv = value[keep]
        return norm, idx, snrv
