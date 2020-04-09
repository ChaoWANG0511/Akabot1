def _bss_decomp_mtifilt(reference_sources, estimated_source, j, flen):

    nsampl = estimated_source.size

    # decomposition

    # true source image

    s_true = np.hstack((reference_sources[j], np.zeros(flen - 1)))

    # spatial (or filtering) distortion

    e_spat = _project(reference_sources[j, np.newaxis, :], estimated_source,

                      flen) - s_true

    # interference

    e_interf = _project(reference_sources,

                        estimated_source, flen) - s_true - e_spat

    # artifacts

    e_artif = -s_true - e_spat - e_interf

    e_artif[:nsampl] += estimated_source

    return (s_true, e_spat, e_interf, e_artif)

def _bss_source_crit(s_true, e_spat, e_interf, e_artif):
    """Measurement of the separation quality for a given source in terms of

    filtered true source, interference and artifacts.

    """

    # energy ratios

    s_filt = s_true + e_spat

    sdr = _safe_db(np.sum(s_filt ** 2), np.sum((e_interf + e_artif) ** 2))

    sir = _safe_db(np.sum(s_filt ** 2), np.sum(e_interf ** 2))

    sar = _safe_db(np.sum((s_filt + e_interf) ** 2), np.sum(e_artif ** 2))

    return (sdr, sir, sar)