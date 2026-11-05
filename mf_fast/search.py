"""The batched short-block search (docs/design-fast.md §5).

For each segment: scale the batch's template spectra to SNR units for that
segment's PSD, cut the segment's series into blocks whose valid windows are
whole 2048-sample bins aligned to ``analyze.start``, and filter every block of
the segment against every template of the batch in one ``run_blocks`` call.
Block bins are stitched back into the segment's PyCBC bin grid, where PyCBC's
symmetric rule is applied; triggers are then clustered across segments per
template with PyCBC's findchirp_cluster_over_window.
"""

import time

import numpy

import matchedfilter as mf
from pycbc.events.eventmgr import findchirp_cluster_over_window

from mf_inspiral.control import symmetric_cluster


def block_layout(analyze, L, R, n, binsize):
    """Block starts and windows for one segment and one template batch.

    The batch's largest support (L before lag 0, R after) sets the valid
    window: the largest whole number of bins with L + valid + R <= n. Block b
    searches segment samples [analyze.start + b*valid, ...), clipped at
    analyze.stop. Returns (starts, win_start, win_end, valid) with windows
    block-local; win_start is L for every block.
    """
    valid = ((n - (L + R + 1)) // binsize) * binsize
    if valid < binsize:
        raise ValueError("support %d + %d does not fit block length %d" % (L, R, n))
    a0, a1 = analyze.start, analyze.stop
    nblk = -(-(a1 - a0) // valid)
    first = a0 + numpy.arange(nblk) * valid
    starts = first - L
    win_start = numpy.full(nblk, L)
    win_end = L + numpy.minimum(valid, a1 - first)
    if starts[0] < 0:
        raise ValueError("first block would start before the segment")
    return starts.astype(numpy.int64), win_start, win_end, valid


def segment_bins(index, value, starts, win_start, win_end, a0, binsize, nbins):
    """Stitch run_blocks output for one template into the segment's bin grid.

    index/value: (nblk, nbins_block) block-local lags and values. Returns
    (seg_index, value) of length nbins, seg_index in segment samples or -1.
    """
    out_i = numpy.full(nbins, -1, numpy.int64)
    out_v = numpy.zeros(nbins, numpy.complex64)
    for b in range(len(starts)):
        g0 = (starts[b] + win_start[b] - a0) // binsize
        nb = -(-(win_end[b] - win_start[b]) // binsize)
        bi = index[b, :nb]
        hit = bi >= 0
        sl = out_i[g0:g0 + nb]
        sl[hit] = starts[b] + bi[hit]
        out_v[g0:g0 + nb][hit] = value[b, :nb][hit]
    return out_i, out_v


def run_blocks_ragged(filt, series, starts, ws, we, binsize, threshold):
    """run_blocks over blocks whose windows may differ in bin count.

    matchedfilter requires one bin count per call, so blocks are grouped by bin
    count (in practice: all full blocks, then the shorter last one), as its own
    automatic layout does. Returns (index, value) of shape (nblk, nt, maxbins),
    with -1 / 0 in the unused slots.
    """
    nb = -(-(we - ws) // binsize)
    out_i = out_v = None
    for count in numpy.unique(nb):
        sel = numpy.nonzero(nb == count)[0]
        i, v = filt.run_blocks(series, starts[sel], ws[sel], we[sel], binsize=binsize,
                               threshold=threshold, raw=True)
        if out_i is None:
            out_i = numpy.full((len(starts), i.shape[1], nb.max()), -1, numpy.int64)
            out_v = numpy.zeros((len(starts), i.shape[1], nb.max()), numpy.complex64)
        out_i[sel, :, :count] = i      # copies: the filter reuses its result buffer
        out_v[sel, :, :count] = v
    return out_i, out_v


class Search(object):
    """Run the peak-only search over all segments for a template cache."""

    def __init__(self, frontend, cache, threshold=3.8, device='cpu',
                 tmpl_batch=64, binsize=2048, log=None):
        self.fe, self.cache = frontend, cache
        self.threshold = float(threshold)
        self.device = device
        self.tmpl_batch = int(tmpl_batch)
        self.binsize = int(binsize)
        self.log = log or (lambda *a: None)
        self.timing = dict(scale=0.0, ingest=0.0, filter=0.0, stitch=0.0, cluster=0.0)
        self.counts = dict(pairs=0, valid_samples=0, calls=0)

    def batches(self):
        """Template batches in order of support, so a batch's L, R are close
        to its members'."""
        order = numpy.argsort(self.cache.support, kind='stable')
        for i in range(0, len(order), self.tmpl_batch):
            yield order[i:i + self.tmpl_batch]

    def run(self):
        fe, c, bs = self.fe, self.cache, self.binsize
        n = c.n
        found = {j: [] for j in range(len(c.rows))}       # cache index -> list of arrays
        filters = {}
        for s, seg in enumerate(fe.segments):
            an = seg.analyze
            nbins = -(-(an.stop - an.start) // bs)
            for batch in self.batches():
                L, R = int(c.L[batch].max()), int(c.R[batch].max())
                starts, ws, we, valid = block_layout(an, L, R, n, bs)
                t = time.perf_counter()
                scale = numpy.array([c.block_scale(j, s) for j in batch],
                                    dtype=numpy.float32)
                spectra = c.spectra[batch] * scale[:, None]
                self.timing['scale'] += time.perf_counter() - t
                key = (len(starts), len(batch))
                filt = filters.get(key)
                if filt is None:
                    filt = filters[key] = mf.MatchedFilter(n, ndata=len(starts),
                                                           ntemplates=len(batch),
                                                           device=self.device)
                t = time.perf_counter()
                filt.set_templates(spectra)
                self.timing['ingest'] += time.perf_counter() - t
                t = time.perf_counter()
                idx, val = run_blocks_ragged(filt, fe.series[s], starts, ws, we, bs,
                                             self.threshold)
                self.timing['filter'] += time.perf_counter() - t
                self.counts['calls'] += 1
                self.counts['pairs'] += len(starts) * len(batch)
                self.counts['valid_samples'] += (an.stop - an.start) * len(batch)
                t = time.perf_counter()
                for k, j in enumerate(batch):
                    si, sv = segment_bins(idx[:, k], val[:, k], starts, ws, we, an.start,
                                          bs, nbins)
                    keep = symmetric_cluster(si, sv, nbins)
                    if keep.any():
                        # time_index as stock: samples from gps_start
                        found[j].append((seg.seg_slice.start + si[keep], sv[keep],
                                         numpy.full(keep.sum(), c.sigmasq[j, s],
                                                    numpy.float32)))
                self.timing['stitch'] += time.perf_counter() - t
            self.log("segment %d/%d done" % (s + 1, len(fe.segments)))
        t = time.perf_counter()
        events = []
        for j in range(len(c.rows)):
            if not found[j]:
                continue
            ti = numpy.concatenate([f[0] for f in found[j]])
            sv = numpy.concatenate([f[1] for f in found[j]])
            sg = numpy.concatenate([f[2] for f in found[j]])
            keep = findchirp_cluster_over_window(ti, sv, bs)
            events.append((j, ti[keep], sv[keep], sg[keep]))
        self.timing['cluster'] += time.perf_counter() - t
        return events
