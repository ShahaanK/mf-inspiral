"""Trigger file in pycbc_inspiral's layout, restricted to the SNR-only core columns.

Columns follow pycbc's EventManager.write_to_hdf (same names, dtypes and gzip-9 +
shuffle compression). There are no chi-squared or PSD-variation columns: mf_fast
isn't a workflow drop-in. Triggers are sorted by bank row, as stock sorts by
template.
"""

import h5py
import numpy

COMPRESSION = dict(compression='gzip', compression_opts=9, shuffle=True)


def write(path, ifo, events, cache, gps_start, sample_rate, trig_start, trig_end,
          extra_attrs=None):
    """events: list of (cache index, time_index, complex snr, sigmasq)."""
    events = sorted(events, key=lambda e: cache.rows[e[0]])
    if events:
        ti = numpy.concatenate([e[1] for e in events])
        snr = numpy.concatenate([e[2] for e in events])
        sg = numpy.concatenate([e[3] for e in events])
        j = numpy.concatenate([numpy.full(len(e[1]), e[0]) for e in events])
    with h5py.File(path, 'w') as f:
        g = f.create_group(ifo)
        if events:
            cols = {
                'snr': numpy.abs(snr).astype(numpy.float32),
                'coa_phase': numpy.angle(snr).astype(numpy.float32),
                'end_time': ti / float(sample_rate) + gps_start,
                'sigmasq': sg.astype(numpy.float32),
                'template_duration': cache.duration[j].astype(numpy.float32),
                'template_hash': cache.hashes[j].astype(numpy.int64),
            }
            for k, v in cols.items():
                g.create_dataset(k, data=v, **COMPRESSION)
        g.create_dataset('search/start_time', data=numpy.array([trig_start]), **COMPRESSION)
        g.create_dataset('search/end_time', data=numpy.array([trig_end]), **COMPRESSION)
        for k, v in (extra_attrs or {}).items():
            f.attrs[k] = v
    return 0 if not events else len(ti)
