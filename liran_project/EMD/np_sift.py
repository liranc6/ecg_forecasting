#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

"""
Low level functionality for the sift algorithm.

  get_padded_extrema
  compute_parabolic_extrema
  interp_envelope
  zero_crossing_count

"""

import logging

import numpy as np

# Housekeeping for logging
logger = logging.getLogger(__name__)


def get_padded_extrema(X, pad_width=2, mode='peaks', parabolic_extrema=False,
                       loc_pad_opts=None, mag_pad_opts=None, method='rilling'):
    """Identify and pad the extrema in a signal.

    This function returns a set of extrema from a signal including padded
    extrema at the edges of the signal. Padding is carried out using numpy.pad.

    Parameters
    ----------
    X : ndarray
        Input signal
    pad_width : int >= 0
        Number of additional extrema to add to the start and end
    mode : {'peaks', 'troughs', 'abs_peaks', 'both'}
        Switch between detecting peaks, troughs, peaks in the abs signal or
        both peaks and troughs
    method : {'rilling', 'numpypad'}
        Which padding method to use
    parabolic_extrema : bool
        Flag indicating whether extrema positions should be refined by parabolic interpolation
    loc_pad_opts : dict
        Optional dictionary of options to be passed to np.pad when padding extrema locations
    mag_pad_opts : dict
        Optional dictionary of options to be passed to np.pad when padding extrema magnitudes

    Returns
    -------
    locs : ndarray
        location of extrema in samples
    mags : ndarray
        Magnitude of each extrema

    See Also
    --------
    emd.sift.interp_envelope
    emd.sift._pad_extrema_numpy
    emd.sift._pad_extrema_rilling

    Notes
    -----
    The 'abs_peaks' mode is not compatible with the 'rilling' method as rilling
    must identify all peaks and troughs together.

    """
    if (mode == 'abs_peaks') and (method == 'rilling'):
        msg = "get_padded_extrema mode 'abs_peaks' is incompatible with method 'rilling'"
        raise ValueError(msg)

    if X.ndim == 2:
        X = X[:, 0]

    if mode == 'both' or method == 'rilling':
        max_locs, max_ext = _find_extrema(X, parabolic_extrema=parabolic_extrema)
        min_locs, min_ext = _find_extrema(-X, parabolic_extrema=parabolic_extrema)
        min_ext = -min_ext
        logger.debug('found {0} minima and {1} maxima on mode {2}'.format(len(min_locs),
                                                                          len(max_locs),
                                                                          mode))
    elif mode == 'peaks':
        max_locs, max_ext = _find_extrema(X, parabolic_extrema=parabolic_extrema)
        logger.debug('found {0} maxima on mode {1}'.format(len(max_locs),
                                                           mode))
    elif mode == 'troughs':
        max_locs, max_ext = _find_extrema(-X, parabolic_extrema=parabolic_extrema)
        max_ext = -max_ext
        logger.debug('found {0} minima on mode {1}'.format(len(max_locs),
                                                           mode))
    elif mode == 'abs_peaks':
        max_locs, max_ext = _find_extrema(np.abs(X), parabolic_extrema=parabolic_extrema)
        logger.debug('found {0} extrema on mode {1}'.format(len(max_locs),
                                                            mode))
    else:
        raise ValueError('Mode {0} not recognised by get_padded_extrema'.format(mode))

    # Return nothing if we don't have enough extrema
    if (len(max_locs) == 0) or (max_locs.size <= 1):
        logger.debug('Not enough extrema to pad.')
        return None, None
    elif (mode == 'both' or method == 'rilling') and len(min_locs) <= 1:
        logger.debug('Not enough extrema to pad 2.')
        return None, None

    # Run the padding by requested method
    if pad_width == 0:
        if mode == 'both':
            ret = (min_locs, min_ext, max_locs, max_ext)
        elif mode == 'troughs' and method == 'rilling':
            ret = (min_locs, min_ext)
        else:
            ret = (max_locs, max_ext)
    elif method == 'numpypad':
        ret = _pad_extrema_numpy(max_locs, max_ext,
                                 X.shape[0], pad_width,
                                 loc_pad_opts, mag_pad_opts)
        if mode == 'both':
            ret2 = _pad_extrema_numpy(min_locs, min_ext,
                                      X.shape[0], pad_width,
                                      loc_pad_opts, mag_pad_opts)
            ret = (ret2[0], ret2[1], ret[0], ret[1])
    elif method == 'rilling':
        ret = _pad_extrema_rilling(min_locs, max_locs, X, pad_width)
        # Inefficient to use rilling for just peaks or troughs, but handle it
        # just in case.
        if mode == 'peaks':
            ret = ret[2:]
        elif mode == 'troughs':
            ret = ret[:2]

    return ret


def _pad_extrema_numpy(locs, mags, lenx, pad_width, loc_pad_opts, mag_pad_opts):
    """Pad extrema using a direct call to np.pad.

    Extra paddings are carried out if the padded values do not span the whole
    range of the original time-series (defined by lenx)

    Parameters
    ----------
    locs : ndarray
        location of extrema in time
    mags : ndarray
        magnitude of each extrema
    lenx : int
        length of the time-series from which locs and mags were identified
    pad_width : int
        number of extra extrema to pad
    loc_pad_opts : dict
        dictionary of argumnents passed to np.pad to generate new extrema locations
    mag_pad_opts : dict
        dictionary of argumnents passed to np.pad to generate new extrema magnitudes

    Returns
    -------
    ndarray
        location of all extrema (including padded and original points) in time
    ndarray
        magnitude of each extrema (including padded and original points)

    """
    logger.verbose("Padding {0} extrema in signal X {1} using method '{2}'".format(pad_width,
                                                                                   lenx,
                                                                                   'numpypad'))

    if not loc_pad_opts:  # Empty dict evaluates to False
        loc_pad_opts = {'mode': 'reflect', 'reflect_type': 'odd'}
    else:
        loc_pad_opts = loc_pad_opts.copy()  # Don't work in place...
    loc_pad_mode = loc_pad_opts.pop('mode')

    if not mag_pad_opts:  # Empty dict evaluates to False
        mag_pad_opts = {'mode': 'median', 'stat_length': 1}
    else:
        mag_pad_opts = mag_pad_opts.copy()  # Don't work in place...
    mag_pad_mode = mag_pad_opts.pop('mode')

    # Determine how much padding to use
    if locs.size < pad_width:
        pad_width = locs.size

    # Return now if we're not padding
    if (pad_width is None) or (pad_width == 0):
        return locs, mags

    # Pad peak locations
    ret_locs = np.pad(locs, pad_width, loc_pad_mode, **loc_pad_opts)

    # Pad peak magnitudes
    ret_mag = np.pad(mags, pad_width, mag_pad_mode, **mag_pad_opts)

    # Keep padding if the locations don't stretch to the edge
    count = 0
    while np.max(ret_locs) < lenx or np.min(ret_locs) >= 0:
        logger.debug('Padding again - first ext {0}, last ext {1}'.format(np.min(ret_locs), np.max(ret_locs)))
        logger.debug(ret_locs)
        ret_locs = np.pad(ret_locs, pad_width, loc_pad_mode, **loc_pad_opts)
        ret_mag = np.pad(ret_mag, pad_width, mag_pad_mode, **mag_pad_opts)
        count += 1
        #if count > 5:
        #    raise ValueError

    return ret_locs, ret_mag


def _pad_extrema_rilling(indmin, indmax, X, pad_width):
    """Pad extrema using the method from Rilling.

    This is based on original matlab code in boundary_conditions_emd.m
    downloaded from: https://perso.ens-lyon.fr/patrick.flandrin/emd.html

    Unlike the numpypad method - this approach pads both the maxima and minima
    of the signal together.

    Parameters
    ----------
    indmin : ndarray
        location of minima in time
    indmax : ndarray
        location of maxima in time
    X : ndarray
        original time-series
    pad_width : int
        number of extra extrema to pad

    Returns
    -------
    tmin
        location of all minima (including padded and original points) in time
    xmin
        magnitude of each minima (including padded and original points)
    tmax
        location of all maxima (including padded and original points) in time
    xmax
        magnitude of each maxima (including padded and original points)

    """
    logger.debug("Padding {0} extrema in signal X {1} using method '{2}'".format(pad_width,
                                                                                 X.shape,
                                                                                 'rilling'))

    t = np.arange(len(X))

    # Pad START
    if indmax[0] < indmin[0]:
        # First maxima is before first minima
        if X[0] > X[indmin[0]]:
            # First value is larger than first minima - reflect about first MAXIMA
            logger.debug('L: max earlier than min, first val larger than first min')
            lmax = np.flipud(indmax[1:pad_width+1])
            lmin = np.flipud(indmin[:pad_width])
            lsym = indmax[0]
        else:
            # First value is smaller than first minima - reflect about first MINIMA
            logger.debug('L: max earlier than min, first val smaller than first min')
            lmax = np.flipud(indmax[:pad_width])
            lmin = np.r_[np.flipud(indmin[:pad_width-1]), 0]
            lsym = 0

    else:
        # First minima is before first maxima
        if X[0] > X[indmax[0]]:
            # First value is larger than first minima - reflect about first MINIMA
            logger.debug('L: max later than min, first val larger than first max')
            lmax = np.flipud(indmax[:pad_width])
            lmin = np.flipud(indmin[1:pad_width+1])
            lsym = indmin[0]
        else:
            # First value is smaller than first minima - reflect about first MAXIMA
            logger.debug('L: max later than min, first val smaller than first max')
            lmin = np.flipud(indmin[:pad_width])
            lmax = np.r_[np.flipud(indmax[:pad_width-1]), 0]
            lsym = 0

    # Pad STOP
    if indmax[-1] < indmin[-1]:
        # Last maxima is before last minima
        if X[-1] < X[indmax[-1]]:
            # Last value is larger than last minima - reflect about first MAXIMA
            logger.debug('R: max earlier than min, last val smaller than last max')
            rmax = np.flipud(indmax[-pad_width:])
            rmin = np.flipud(indmin[-pad_width-1:-1])
            rsym = indmin[-1]
        else:
            # First value is smaller than first minima - reflect about first MINIMA
            logger.debug('R: max earlier than min, last val larger than last max')
            rmax = np.r_[X.shape[0] - 1, np.flipud(indmax[-(pad_width-2):])]
            rmin = np.flipud(indmin[-(pad_width-1):])
            rsym = X.shape[0] - 1

    else:
        if X[-1] > X[indmin[-1]]:
            # Last value is larger than last minima - reflect about first MAXIMA
            logger.debug('R: max later than min, last val larger than last min')
            rmax = np.flipud(indmax[-pad_width-1:-1])
            rmin = np.flipud(indmin[-pad_width:])
            rsym = indmax[-1]
        else:
            # First value is smaller than first minima - reflect about first MINIMA
            logger.debug('R: max later than min, last val smaller than last min')
            rmax = np.flipud(indmax[-(pad_width-1):])
            rmin = np.r_[X.shape[0] - 1, np.flipud(indmin[-(pad_width-2):])]
            rsym = X.shape[0] - 1

    # Extrema values are ordered from largest to smallest,
    # lmin and lmax are the samples of the first {pad_width} extrema
    # rmin and rmax are the samples of the final {pad_width} extrema

    # Compute padded samples
    tlmin = 2 * lsym - lmin
    tlmax = 2 * lsym - lmax
    trmin = 2 * rsym - rmin
    trmax = 2 * rsym - rmax

    # tlmin and tlmax are the samples of the left/first padded extrema, in ascending order
    # trmin and trmax are the samples of the right/final padded extrema, in ascending order

    # Flip again if needed - don't really get what this is doing, will trust the source...
    if (tlmin[0] >= t[0]) or (tlmax[0] >= t[0]):
        msg = 'Flipping start again - first min: {0}, first max: {1}, t[0]: {2}'
        logger.debug(msg.format(tlmin[0], tlmax[0], t[0]))
        if lsym == indmax[0]:
            lmax = np.flipud(indmax[:pad_width])
        else:
            lmin = np.flipud(indmin[:pad_width])
        lsym = 0
        tlmin = 2*lsym-lmin
        tlmax = 2*lsym-lmax

        if tlmin[0] >= t[0]:
            raise ValueError('Left min not padded enough. {0} {1}'.format(tlmin[0], t[0]))
        if tlmax[0] >= t[0]:
            raise ValueError('Left max not padded enough. {0} {1}'.format(trmax[0], t[0]))

    if (trmin[-1] <= t[-1]) or (trmax[-1] <= t[-1]):
        msg = 'Flipping end again - last min: {0}, last max: {1}, t[-1]: {2}'
        logger.debug(msg.format(trmin[-1], trmax[-1], t[-1]))
        if rsym == indmax[-1]:
            rmax = np.flipud(indmax[-pad_width-1:-1])
        else:
            rmin = np.flipud(indmin[-pad_width-1:-1])
        rsym = len(X)
        trmin = 2*rsym-rmin
        trmax = 2*rsym-rmax

        if trmin[-1] <= t[-1]:
            raise ValueError('Right min not padded enough. {0} {1}'.format(trmin[-1], t[-1]))
        if trmax[-1] <= t[-1]:
            raise ValueError('Right max not padded enough. {0} {1}'.format(trmax[-1], t[-1]))

    # Stack and return padded values
    ret_tmin = np.r_[tlmin, t[indmin], trmin]
    ret_tmax = np.r_[tlmax, t[indmax], trmax]

    ret_xmin = np.r_[X[lmin], X[indmin], X[rmin]]
    ret_xmax = np.r_[X[lmax], X[indmax], X[rmax]]

    # Quick check that interpolation won't explode
    if np.all(np.diff(ret_tmin) > 0) is False:
        logger.warning('Minima locations not strictly ascending - interpolation will break')
        raise ValueError('Extrema locations not strictly ascending!!')
    if np.all(np.diff(ret_tmax) > 0) is False:
        logger.warning('Maxima locations not strictly ascending - interpolation will break')
        raise ValueError('Extrema locations not strictly ascending!!')

    return ret_tmin, ret_xmin, ret_tmax, ret_xmax


def _find_extrema(X, peak_prom_thresh=None, parabolic_extrema=False):
    """Identify extrema within a time-course.

    This function detects extrema using a scipy.signals.argrelextrema. Extrema
    locations can be refined by parabolic intpolation and optionally
    thresholded by peak prominence.

    Parameters
    ----------
    X : ndarray
       Input signal
    peak_prom_thresh : {None, float}
       Only include peaks which have prominences above this threshold or None
       for no threshold (default is no threshold)
    parabolic_extrema : bool
        Flag indicating whether peak estimation should be refined by parabolic
        interpolation (default is False)

    Returns
    -------
    locs : ndarray
        Location of extrema in samples
    extrema : ndarray
        Value of each extrema

    """
    from scipy.signal import argrelextrema
    ext_locs = argrelextrema(X, np.greater, order=1)[0]

    if len(ext_locs) == 0:
        return np.array([]), np.array([])

    from scipy.signal._peak_finding import peak_prominences
    if peak_prom_thresh is not None:
        prom, _, _ = peak_prominences(X, ext_locs, wlen=3)
        keeps = np.where(prom > peak_prom_thresh)[0]
        ext_locs = ext_locs[keeps]

    if parabolic_extrema:
        y = np.c_[X[ext_locs-1], X[ext_locs], X[ext_locs+1]].T
        ext_locs, max_pks = compute_parabolic_extrema(y, ext_locs)
        return ext_locs, max_pks
    else:
        return ext_locs, X[ext_locs]


def compute_parabolic_extrema(y, locs):
    """Compute a parabolic refinement extrema locations.

    Parabolic refinement is computed from in triplets of points based on the
    method described in section 3.2.1 from Rato 2008 [1]_.

    Parameters
    ----------
    y : array_like
        A [3 x nextrema] array containing the points immediately around the
        extrema in a time-series.
    locs : array_like
        A [nextrema] length vector containing x-axis positions of the extrema

    Returns
    -------
    numpy array
        The estimated y-axis values of the interpolated extrema
    numpy array
        The estimated x-axis values of the interpolated extrema

    References
    ----------
    .. [1] Rato, R. T., Ortigueira, M. D., & Batista, A. G. (2008). On the HHT,
    its problems, and some solutions. Mechanical Systems and Signal Processing,
    22(6), 1374–1394. https://doi.org/10.1016/j.ymssp.2007.11.028

    """
    # Parabola equation parameters for computing y from parameters a, b and c
    # w = np.array([[1, 1, 1], [4, 2, 1], [9, 3, 1]])
    # ... and its inverse for computing a, b and c from y
    w_inv = np.array([[.5, -1, .5], [-5/2, 4, -3/2], [3, -3, 1]])
    abc = w_inv.dot(y)

    # Find co-ordinates of extrema from parameters abc
    tp = - abc[1, :] / (2*abc[0, :])
    t = tp - 2 + locs
    y_hat = tp*abc[1, :]/2 + abc[2, :]

    return t, y_hat


def interp_envelope(X, mode='both', interp_method='splrep', extrema_opts=None,
                    ret_extrema=False, trim=True):
    """Interpolate the amplitude envelope of a signal.

    Parameters
    ----------
    X : ndarray
        Input signal
    mode : {'upper','lower','combined'}
         Flag to set which envelope should be computed (Default value = 'upper')
    interp_method : {'splrep','pchip','mono_pchip'}
         Flag to indicate which interpolation method should be used (Default value = 'splrep')

    Returns
    -------
    ndarray
        Interpolated amplitude envelope

    """
    if not extrema_opts:  # Empty dict evaluates to False
        extrema_opts = {'pad_width': 2,
                        'loc_pad_opts': None,
                        'mag_pad_opts': None}
    else:
        extrema_opts = extrema_opts.copy()  # Don't work in place...

    logger.debug("Interpolating '{0}' with method '{1}'".format(mode, interp_method))

    if interp_method not in ['splrep', 'mono_pchip', 'pchip']:
        raise ValueError("Invalid interp_method value")

    if mode == 'upper':
        extr = get_padded_extrema(X, mode='peaks', **extrema_opts)
    elif mode == 'lower':
        extr = get_padded_extrema(X, mode='troughs', **extrema_opts)
    elif (mode == 'both') or (extrema_opts.get('method', '') == 'rilling'):
        extr = get_padded_extrema(X, mode='both', **extrema_opts)
    elif mode == 'combined':
        extr = get_padded_extrema(X, mode='abs_peaks', **extrema_opts)
    else:
        raise ValueError('Mode not recognised. Use mode= \'upper\'|\'lower\'|\'combined\'')

    if extr[0] is None:
        if mode == 'both':
            return None, None
        else:
            return None

    if mode == 'both':
        lower = _run_scipy_interp(extr[0], extr[1],
                                  lenx=X.shape[0], trim=trim,
                                  interp_method=interp_method)
        upper = _run_scipy_interp(extr[2], extr[3],
                                  lenx=X.shape[0], trim=trim,
                                  interp_method=interp_method)
        env = (upper, lower)
    else:
        env = _run_scipy_interp(extr[0], extr[1], lenx=X.shape[0], interp_method=interp_method, trim=trim)

    if ret_extrema:
        return env, extr
    else:
        return env


def _run_scipy_interp(locs, pks, lenx, interp_method='splrep', trim=True):
    from scipy import interpolate as interp

    # Run interpolation on envelope
    t = np.arange(locs[0], locs[-1])
    if interp_method == 'splrep':
        f = interp.splrep(locs, pks)
        env = interp.splev(t, f)
    elif interp_method == 'mono_pchip':
        pchip = interp.PchipInterpolator(locs, pks)
        env = pchip(t)
    elif interp_method == 'pchip':
        pchip = interp.pchip(locs, pks)
        env = pchip(t)

    if trim:
        t_max = np.arange(locs[0], locs[-1])
        tinds = np.logical_and((t_max >= 0), (t_max < lenx))
        env = np.array(env[tinds])

        if env.shape[0] != lenx:
            msg = 'Envelope length does not match input data {0} {1}'
            raise ValueError(msg.format(env.shape[0], lenx))

    return env


def zero_crossing_count(X):
    """Count the number of zero-crossings within a time-course.

    Zero-crossings are counted through differentiation of the sign of the
    signal.

    Parameters
    ----------
    X : ndarray
        Input array

    Returns
    -------
    int
        Number of zero-crossings

    """
    if X.ndim == 2:
        X = X[:, None]

    return (np.diff(np.sign(X), axis=0) != 0).sum(axis=0)

#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

"""
Low level functionality for the sift algorithm.

  get_padded_extrema
  compute_parabolic_extrema
  interp_envelope
  zero_crossing_count

"""

# Housekeeping for logging
logger = logging.getLogger(__name__)


def get_padded_extrema(X, pad_width=2, mode='peaks', parabolic_extrema=False,
                       loc_pad_opts=None, mag_pad_opts=None, method='rilling'):
    """Identify and pad the extrema in a signal.

    This function returns a set of extrema from a signal including padded
    extrema at the edges of the signal. Padding is carried out using numpy.pad.

    Parameters
    ----------
    X : ndarray
        Input signal
    pad_width : int >= 0
        Number of additional extrema to add to the start and end
    mode : {'peaks', 'troughs', 'abs_peaks', 'both'}
        Switch between detecting peaks, troughs, peaks in the abs signal or
        both peaks and troughs
    method : {'rilling', 'numpypad'}
        Which padding method to use
    parabolic_extrema : bool
        Flag indicating whether extrema positions should be refined by parabolic interpolation
    loc_pad_opts : dict
        Optional dictionary of options to be passed to np.pad when padding extrema locations
    mag_pad_opts : dict
        Optional dictionary of options to be passed to np.pad when padding extrema magnitudes

    Returns
    -------
    locs : ndarray
        location of extrema in samples
    mags : ndarray
        Magnitude of each extrema

    See Also
    --------
    emd.sift.interp_envelope
    emd.sift._pad_extrema_numpy
    emd.sift._pad_extrema_rilling

    Notes
    -----
    The 'abs_peaks' mode is not compatible with the 'rilling' method as rilling
    must identify all peaks and troughs together.

    """
    if (mode == 'abs_peaks') and (method == 'rilling'):
        msg = "get_padded_extrema mode 'abs_peaks' is incompatible with method 'rilling'"
        raise ValueError(msg)

    if X.ndim == 2:
        X = X[:, 0]

    if mode == 'both' or method == 'rilling':
        max_locs, max_ext = _find_extrema(X, parabolic_extrema=parabolic_extrema)
        min_locs, min_ext = _find_extrema(-X, parabolic_extrema=parabolic_extrema)
        min_ext = -min_ext
        logger.debug('found {0} minima and {1} maxima on mode {2}'.format(len(min_locs),
                                                                          len(max_locs),
                                                                          mode))
    elif mode == 'peaks':
        max_locs, max_ext = _find_extrema(X, parabolic_extrema=parabolic_extrema)
        logger.debug('found {0} maxima on mode {1}'.format(len(max_locs),
                                                           mode))
    elif mode == 'troughs':
        max_locs, max_ext = _find_extrema(-X, parabolic_extrema=parabolic_extrema)
        max_ext = -max_ext
        logger.debug('found {0} minima on mode {1}'.format(len(max_locs),
                                                           mode))
    elif mode == 'abs_peaks':
        max_locs, max_ext = _find_extrema(np.abs(X), parabolic_extrema=parabolic_extrema)
        logger.debug('found {0} extrema on mode {1}'.format(len(max_locs),
                                                            mode))
    else:
        raise ValueError('Mode {0} not recognised by get_padded_extrema'.format(mode))

    # Return nothing if we don't have enough extrema
    if (len(max_locs) == 0) or (max_locs.size <= 1):
        logger.debug('Not enough extrema to pad.')
        return None, None
    elif (mode == 'both' or method == 'rilling') and len(min_locs) <= 1:
        logger.debug('Not enough extrema to pad 2.')
        return None, None

    # Run the padding by requested method
    if pad_width == 0:
        if mode == 'both':
            ret = (min_locs, min_ext, max_locs, max_ext)
        elif mode == 'troughs' and method == 'rilling':
            ret = (min_locs, min_ext)
        else:
            ret = (max_locs, max_ext)
    elif method == 'numpypad':
        ret = _pad_extrema_numpy(max_locs, max_ext,
                                 X.shape[0], pad_width,
                                 loc_pad_opts, mag_pad_opts)
        if mode == 'both':
            ret2 = _pad_extrema_numpy(min_locs, min_ext,
                                      X.shape[0], pad_width,
                                      loc_pad_opts, mag_pad_opts)
            ret = (ret2[0], ret2[1], ret[0], ret[1])
    elif method == 'rilling':
        ret = _pad_extrema_rilling(min_locs, max_locs, X, pad_width)
        # Inefficient to use rilling for just peaks or troughs, but handle it
        # just in case.
        if mode == 'peaks':
            ret = ret[2:]
        elif mode == 'troughs':
            ret = ret[:2]

    return ret


def _pad_extrema_numpy(locs, mags, lenx, pad_width, loc_pad_opts, mag_pad_opts):
    """Pad extrema using a direct call to np.pad.

    Extra paddings are carried out if the padded values do not span the whole
    range of the original time-series (defined by lenx)

    Parameters
    ----------
    locs : ndarray
        location of extrema in time
    mags : ndarray
        magnitude of each extrema
    lenx : int
        length of the time-series from which locs and mags were identified
    pad_width : int
        number of extra extrema to pad
    loc_pad_opts : dict
        dictionary of argumnents passed to np.pad to generate new extrema locations
    mag_pad_opts : dict
        dictionary of argumnents passed to np.pad to generate new extrema magnitudes

    Returns
    -------
    ndarray
        location of all extrema (including padded and original points) in time
    ndarray
        magnitude of each extrema (including padded and original points)

    """
    logger.verbose("Padding {0} extrema in signal X {1} using method '{2}'".format(pad_width,
                                                                                   lenx,
                                                                                   'numpypad'))

    if not loc_pad_opts:  # Empty dict evaluates to False
        loc_pad_opts = {'mode': 'reflect', 'reflect_type': 'odd'}
    else:
        loc_pad_opts = loc_pad_opts.copy()  # Don't work in place...
    loc_pad_mode = loc_pad_opts.pop('mode')

    if not mag_pad_opts:  # Empty dict evaluates to False
        mag_pad_opts = {'mode': 'median', 'stat_length': 1}
    else:
        mag_pad_opts = mag_pad_opts.copy()  # Don't work in place...
    mag_pad_mode = mag_pad_opts.pop('mode')

    # Determine how much padding to use
    if locs.size < pad_width:
        pad_width = locs.size

    # Return now if we're not padding
    if (pad_width is None) or (pad_width == 0):
        return locs, mags

    # Pad peak locations
    ret_locs = np.pad(locs, pad_width, loc_pad_mode, **loc_pad_opts)

    # Pad peak magnitudes
    ret_mag = np.pad(mags, pad_width, mag_pad_mode, **mag_pad_opts)

    # Keep padding if the locations don't stretch to the edge
    count = 0
    while np.max(ret_locs) < lenx or np.min(ret_locs) >= 0:
        logger.debug('Padding again - first ext {0}, last ext {1}'.format(np.min(ret_locs), np.max(ret_locs)))
        logger.debug(ret_locs)
        ret_locs = np.pad(ret_locs, pad_width, loc_pad_mode, **loc_pad_opts)
        ret_mag = np.pad(ret_mag, pad_width, mag_pad_mode, **mag_pad_opts)
        count += 1
        #if count > 5:
        #    raise ValueError

    return ret_locs, ret_mag


def _pad_extrema_rilling(indmin, indmax, X, pad_width):
    """Pad extrema using the method from Rilling.

    This is based on original matlab code in boundary_conditions_emd.m
    downloaded from: https://perso.ens-lyon.fr/patrick.flandrin/emd.html

    Unlike the numpypad method - this approach pads both the maxima and minima
    of the signal together.

    Parameters
    ----------
    indmin : ndarray
        location of minima in time
    indmax : ndarray
        location of maxima in time
    X : ndarray
        original time-series
    pad_width : int
        number of extra extrema to pad

    Returns
    -------
    tmin
        location of all minima (including padded and original points) in time
    xmin
        magnitude of each minima (including padded and original points)
    tmax
        location of all maxima (including padded and original points) in time
    xmax
        magnitude of each maxima (including padded and original points)

    """
    logger.debug("Padding {0} extrema in signal X {1} using method '{2}'".format(pad_width,
                                                                                 X.shape,
                                                                                 'rilling'))

    t = np.arange(len(X))

    # Pad START
    if indmax[0] < indmin[0]:
        # First maxima is before first minima
        if X[0] > X[indmin[0]]:
            # First value is larger than first minima - reflect about first MAXIMA
            logger.debug('L: max earlier than min, first val larger than first min')
            lmax = np.flipud(indmax[1:pad_width+1])
            lmin = np.flipud(indmin[:pad_width])
            lsym = indmax[0]
        else:
            # First value is smaller than first minima - reflect about first MINIMA
            logger.debug('L: max earlier than min, first val smaller than first min')
            lmax = np.flipud(indmax[:pad_width])
            lmin = np.r_[np.flipud(indmin[:pad_width-1]), 0]
            lsym = 0

    else:
        # First minima is before first maxima
        if X[0] > X[indmax[0]]:
            # First value is larger than first minima - reflect about first MINIMA
            logger.debug('L: max later than min, first val larger than first max')
            lmax = np.flipud(indmax[:pad_width])
            lmin = np.flipud(indmin[1:pad_width+1])
            lsym = indmin[0]
        else:
            # First value is smaller than first minima - reflect about first MAXIMA
            logger.debug('L: max later than min, first val smaller than first max')
            lmin = np.flipud(indmin[:pad_width])
            lmax = np.r_[np.flipud(indmax[:pad_width-1]), 0]
            lsym = 0

    # Pad STOP
    if indmax[-1] < indmin[-1]:
        # Last maxima is before last minima
        if X[-1] < X[indmax[-1]]:
            # Last value is larger than last minima - reflect about first MAXIMA
            logger.debug('R: max earlier than min, last val smaller than last max')
            rmax = np.flipud(indmax[-pad_width:])
            rmin = np.flipud(indmin[-pad_width-1:-1])
            rsym = indmin[-1]
        else:
            # First value is smaller than first minima - reflect about first MINIMA
            logger.debug('R: max earlier than min, last val larger than last max')
            rmax = np.r_[X.shape[0] - 1, np.flipud(indmax[-(pad_width-2):])]
            rmin = np.flipud(indmin[-(pad_width-1):])
            rsym = X.shape[0] - 1

    else:
        if X[-1] > X[indmin[-1]]:
            # Last value is larger than last minima - reflect about first MAXIMA
            logger.debug('R: max later than min, last val larger than last min')
            rmax = np.flipud(indmax[-pad_width-1:-1])
            rmin = np.flipud(indmin[-pad_width:])
            rsym = indmax[-1]
        else:
            # First value is smaller than first minima - reflect about first MINIMA
            logger.debug('R: max later than min, last val smaller than last min')
            rmax = np.flipud(indmax[-(pad_width-1):])
            rmin = np.r_[X.shape[0] - 1, np.flipud(indmin[-(pad_width-2):])]
            rsym = X.shape[0] - 1

    # Extrema values are ordered from largest to smallest,
    # lmin and lmax are the samples of the first {pad_width} extrema
    # rmin and rmax are the samples of the final {pad_width} extrema

    # Compute padded samples
    tlmin = 2 * lsym - lmin
    tlmax = 2 * lsym - lmax
    trmin = 2 * rsym - rmin
    trmax = 2 * rsym - rmax

    # tlmin and tlmax are the samples of the left/first padded extrema, in ascending order
    # trmin and trmax are the samples of the right/final padded extrema, in ascending order

    # Flip again if needed - don't really get what this is doing, will trust the source...
    if (tlmin[0] >= t[0]) or (tlmax[0] >= t[0]):
        msg = 'Flipping start again - first min: {0}, first max: {1}, t[0]: {2}'
        logger.debug(msg.format(tlmin[0], tlmax[0], t[0]))
        if lsym == indmax[0]:
            lmax = np.flipud(indmax[:pad_width])
        else:
            lmin = np.flipud(indmin[:pad_width])
        lsym = 0
        tlmin = 2*lsym-lmin
        tlmax = 2*lsym-lmax

        if tlmin[0] >= t[0]:
            raise ValueError('Left min not padded enough. {0} {1}'.format(tlmin[0], t[0]))
        if tlmax[0] >= t[0]:
            raise ValueError('Left max not padded enough. {0} {1}'.format(trmax[0], t[0]))

    if (trmin[-1] <= t[-1]) or (trmax[-1] <= t[-1]):
        msg = 'Flipping end again - last min: {0}, last max: {1}, t[-1]: {2}'
        logger.debug(msg.format(trmin[-1], trmax[-1], t[-1]))
        if rsym == indmax[-1]:
            rmax = np.flipud(indmax[-pad_width-1:-1])
        else:
            rmin = np.flipud(indmin[-pad_width-1:-1])
        rsym = len(X)
        trmin = 2*rsym-rmin
        trmax = 2*rsym-rmax

        if trmin[-1] <= t[-1]:
            raise ValueError('Right min not padded enough. {0} {1}'.format(trmin[-1], t[-1]))
        if trmax[-1] <= t[-1]:
            raise ValueError('Right max not padded enough. {0} {1}'.format(trmax[-1], t[-1]))

    # Stack and return padded values
    ret_tmin = np.r_[tlmin, t[indmin], trmin]
    ret_tmax = np.r_[tlmax, t[indmax], trmax]

    ret_xmin = np.r_[X[lmin], X[indmin], X[rmin]]
    ret_xmax = np.r_[X[lmax], X[indmax], X[rmax]]

    # Quick check that interpolation won't explode
    if np.all(np.diff(ret_tmin) > 0) is False:
        logger.warning('Minima locations not strictly ascending - interpolation will break')
        raise ValueError('Extrema locations not strictly ascending!!')
    if np.all(np.diff(ret_tmax) > 0) is False:
        logger.warning('Maxima locations not strictly ascending - interpolation will break')
        raise ValueError('Extrema locations not strictly ascending!!')

    return ret_tmin, ret_xmin, ret_tmax, ret_xmax


def _find_extrema(X, peak_prom_thresh=None, parabolic_extrema=False):
    """Identify extrema within a time-course.

    This function detects extrema using a scipy.signals.argrelextrema. Extrema
    locations can be refined by parabolic intpolation and optionally
    thresholded by peak prominence.

    Parameters
    ----------
    X : ndarray
       Input signal
    peak_prom_thresh : {None, float}
       Only include peaks which have prominences above this threshold or None
       for no threshold (default is no threshold)
    parabolic_extrema : bool
        Flag indicating whether peak estimation should be refined by parabolic
        interpolation (default is False)

    Returns
    -------
    locs : ndarray
        Location of extrema in samples
    extrema : ndarray
        Value of each extrema

    """
    from scipy.signal import argrelextrema
    ext_locs = argrelextrema(X, np.greater, order=1)[0]

    if len(ext_locs) == 0:
        return np.array([]), np.array([])

    from scipy.signal._peak_finding import peak_prominences
    if peak_prom_thresh is not None:
        prom, _, _ = peak_prominences(X, ext_locs, wlen=3)
        keeps = np.where(prom > peak_prom_thresh)[0]
        ext_locs = ext_locs[keeps]

    if parabolic_extrema:
        y = np.c_[X[ext_locs-1], X[ext_locs], X[ext_locs+1]].T
        ext_locs, max_pks = compute_parabolic_extrema(y, ext_locs)
        return ext_locs, max_pks
    else:
        return ext_locs, X[ext_locs]


def compute_parabolic_extrema(y, locs):
    """Compute a parabolic refinement extrema locations.

    Parabolic refinement is computed from in triplets of points based on the
    method described in section 3.2.1 from Rato 2008 [1]_.

    Parameters
    ----------
    y : array_like
        A [3 x nextrema] array containing the points immediately around the
        extrema in a time-series.
    locs : array_like
        A [nextrema] length vector containing x-axis positions of the extrema

    Returns
    -------
    numpy array
        The estimated y-axis values of the interpolated extrema
    numpy array
        The estimated x-axis values of the interpolated extrema

    References
    ----------
    .. [1] Rato, R. T., Ortigueira, M. D., & Batista, A. G. (2008). On the HHT,
    its problems, and some solutions. Mechanical Systems and Signal Processing,
    22(6), 1374–1394. https://doi.org/10.1016/j.ymssp.2007.11.028

    """
    # Parabola equation parameters for computing y from parameters a, b and c
    # w = np.array([[1, 1, 1], [4, 2, 1], [9, 3, 1]])
    # ... and its inverse for computing a, b and c from y
    w_inv = np.array([[.5, -1, .5], [-5/2, 4, -3/2], [3, -3, 1]])
    abc = w_inv.dot(y)

    # Find co-ordinates of extrema from parameters abc
    tp = - abc[1, :] / (2*abc[0, :])
    t = tp - 2 + locs
    y_hat = tp*abc[1, :]/2 + abc[2, :]

    return t, y_hat


def interp_envelope(X, mode='both', interp_method='splrep', extrema_opts=None,
                    ret_extrema=False, trim=True):
    """Interpolate the amplitude envelope of a signal.

    Parameters
    ----------
    X : ndarray
        Input signal
    mode : {'upper','lower','combined'}
         Flag to set which envelope should be computed (Default value = 'upper')
    interp_method : {'splrep','pchip','mono_pchip'}
         Flag to indicate which interpolation method should be used (Default value = 'splrep')

    Returns
    -------
    ndarray
        Interpolated amplitude envelope

    """
    if not extrema_opts:  # Empty dict evaluates to False
        extrema_opts = {'pad_width': 2,
                        'loc_pad_opts': None,
                        'mag_pad_opts': None}
    else:
        extrema_opts = extrema_opts.copy()  # Don't work in place...

    logger.debug("Interpolating '{0}' with method '{1}'".format(mode, interp_method))

    if interp_method not in ['splrep', 'mono_pchip', 'pchip']:
        raise ValueError("Invalid interp_method value")

    if mode == 'upper':
        extr = get_padded_extrema(X, mode='peaks', **extrema_opts)
    elif mode == 'lower':
        extr = get_padded_extrema(X, mode='troughs', **extrema_opts)
    elif (mode == 'both') or (extrema_opts.get('method', '') == 'rilling'):
        extr = get_padded_extrema(X, mode='both', **extrema_opts)
    elif mode == 'combined':
        extr = get_padded_extrema(X, mode='abs_peaks', **extrema_opts)
    else:
        raise ValueError('Mode not recognised. Use mode= \'upper\'|\'lower\'|\'combined\'')

    if extr[0] is None:
        if mode == 'both':
            return None, None
        else:
            return None

    if mode == 'both':
        lower = _run_scipy_interp(extr[0], extr[1],
                                  lenx=X.shape[0], trim=trim,
                                  interp_method=interp_method)
        upper = _run_scipy_interp(extr[2], extr[3],
                                  lenx=X.shape[0], trim=trim,
                                  interp_method=interp_method)
        env = (upper, lower)
    else:
        env = _run_scipy_interp(extr[0], extr[1], lenx=X.shape[0], interp_method=interp_method, trim=trim)

    if ret_extrema:
        return env, extr
    else:
        return env


def _run_scipy_interp(locs, pks, lenx, interp_method='splrep', trim=True):
    from scipy import interpolate as interp

    # Run interpolation on envelope
    t = np.arange(locs[0], locs[-1])
    if interp_method == 'splrep':
        f = interp.splrep(locs, pks)
        env = interp.splev(t, f)
    elif interp_method == 'mono_pchip':
        pchip = interp.PchipInterpolator(locs, pks)
        env = pchip(t)
    elif interp_method == 'pchip':
        pchip = interp.pchip(locs, pks)
        env = pchip(t)

    if trim:
        t_max = np.arange(locs[0], locs[-1])
        tinds = np.logical_and((t_max >= 0), (t_max < lenx))
        env = np.array(env[tinds])

        if env.shape[0] != lenx:
            msg = 'Envelope length does not match input data {0} {1}'
            raise ValueError(msg.format(env.shape[0], lenx))

    return env


def zero_crossing_count(X):
    """Count the number of zero-crossings within a time-course.

    Zero-crossings are counted through differentiation of the sign of the
    signal.

    Parameters
    ----------
    X : ndarray
        Input array

    Returns
    -------
    int
        Number of zero-crossings

    """
    if X.ndim == 2:
        X = X[:, None]

    return (np.diff(np.sign(X), axis=0) != 0).sum(axis=0)

def sift(X, sift_thresh=1e-8, energy_thresh=50, rilling_thresh=None,
         max_imfs=None, verbose=None, return_residual=True,
         imf_opts=None, envelope_opts=None, extrema_opts=None):
    """Compute Intrinsic Mode Functions from an input data vector.

    This function implements the original sift algorithm [1]_.

    Parameters
    ----------
    X : ndarray
        1D input array containing the time-series data to be decomposed
    sift_thresh : float
         The threshold at which the overall sifting process will stop. (Default value = 1e-8)
    max_imfs : int
         The maximum number of IMFs to compute. (Default value = None)

    Returns
    -------
    imf: ndarray
        2D array [samples x nimfs] containing he Intrisic Mode Functions from the decomposition of X.

    Other Parameters
    ----------------
    imf_opts : dict
        Optional dictionary of keyword options to be passed to emd.get_next_imf
    envelope_opts : dict
        Optional dictionary of keyword options to be passed to emd.interp_envelope
    extrema_opts : dict
        Optional dictionary of keyword options to be passed to emd.get_padded_extrema
    verbose : {None,'CRITICAL','WARNING','INFO','DEBUG'}
        Option to override the EMD logger level for a call to this function.

    See Also
    --------
    emd.sift.get_next_imf
    emd.sift.get_config

    Notes
    -----
    The classic sift is computed by passing an input vector with all options
    left to default

    >>> imf = emd.sift.sift(x)

    The sift can be customised by passing additional options, here we only
    compute the first four IMFs.

    >>> imf = emd.sift.sift(x, max_imfs=4)

    More detailed options are passed as dictionaries which are passed to the
    relevant lower-level functions. For instance `imf_opts` are passed to
    `get_next_imf`.

    >>> imf_opts = {'env_step_size': 1/3, 'stop_method': 'rilling'}
    >>> imf = emd.sift.sift(x, max_imfs=4, imf_opts=imf_opts)

    A modified dictionary of all options can be created using `get_config`.
    This can be modified and used by unpacking the options into a `sift` call.

    >>> conf = emd.sift.get_config('sift')
    >>> conf['max_imfs'] = 4
    >>> conf['imf_opts'] = imf_opts
    >>> imfs = emd.sift.sift(x, **conf)

    References
    ----------
    .. [1] Huang, N. E., Shen, Z., Long, S. R., Wu, M. C., Shih, H. H., Zheng,
       Q., … Liu, H. H. (1998). The empirical mode decomposition and the Hilbert
       spectrum for nonlinear and non-stationary time series analysis. Proceedings
       of the Royal Society of London. Series A: Mathematical, Physical and
       Engineering Sciences, 454(1971), 903–995.
       https://doi.org/10.1098/rspa.1998.0193

    """
    if not imf_opts:
        imf_opts = {'env_step_size': 1,
                    'sd_thresh': .1}
        
    X = ensure_1d_with_singleton([X], ['X'], 'sift')

    _nsamples_warn(X.shape[0], max_imfs)

    layer = 0
    # Only evaluate peaks and if already an IMF if rilling is specified.
    continue_sift = check_sift_continue(X, X, layer,
                                        max_imfs=max_imfs,
                                        sift_thresh=None,
                                        energy_thresh=None,
                                        rilling_thresh=None,
                                        envelope_opts=envelope_opts,
                                        extrema_opts=extrema_opts,
                                        merge_tests=True)

    proto_imf = X.copy()

    while continue_sift:

        logger.info('sifting IMF : {0}'.format(layer))

        next_imf, continue_sift = get_next_imf(proto_imf,
                                               envelope_opts=envelope_opts,
                                               extrema_opts=extrema_opts,
                                               **imf_opts)

        if layer == 0:
            imf = next_imf
        else:
            imf = np.concatenate((imf, next_imf), axis=1)
            
        
        # print(imf.squeeze())    

        proto_imf = X - imf.sum(axis=1)[:, None]
        layer += 1

        # Check if sifting should continue - all metrics whose thresh is not
        # None will be assessed and sifting will stop if any metric says so
        continue_sift = check_sift_continue(X, proto_imf, layer,
                                            max_imfs=max_imfs,
                                            sift_thresh=sift_thresh,
                                            energy_thresh=energy_thresh,
                                            rilling_thresh=None,
                                            envelope_opts=envelope_opts,
                                            extrema_opts=extrema_opts,
                                            merge_tests=True)

    # Append final residual as last mode - unless its empty
    if np.sum(np.abs(proto_imf)) != 0:
        imf = np.c_[imf, proto_imf]

    return imf

def _set_rilling_defaults(rilling_thresh):
    rilling_thresh = (0.05, 0.5, 0.05) if rilling_thresh is True else rilling_thresh
    return rilling_thresh


def ensure_1d_with_singleton(to_check, names, func_name):
    """Check that a set of arrays are all vectors with singleton second dimensions.

    1d arrays will have a singleton second dimension added and an error will be
    raised for non-singleton 2d or greater than 2d inputs.

    Parameters
    ----------
    to_check : list of arrays
        List of arrays to check for equal dimensions
    names : list
        List of variable names for arrays in to_check
    func_name : str
        Name of function calling ensure_equal_dims

    Returns
    -------
    out
        Copy of arrays in to_check with '1d with singleton' shape.

    Raises
    ------
    ValueError
        If any input is a 2d or greater array

    """
    out_args = list(to_check)
    for idx, xx in enumerate(to_check):

        if (xx.ndim >= 2) and np.all(xx.shape[1:] == np.ones_like(xx.shape[1:])):
            # nd input where all trailing are ones
            msg = "Checking {0} inputs - Trimming trailing singletons from input '{1}' (input size {2})"
            logger.debug(msg.format(func_name, names[idx], xx.shape))
            out_args[idx] = np.squeeze(xx)[:, np.newaxis]
        elif (xx.ndim >= 2) and np.all(xx.shape[1:] == np.ones_like(xx.shape[1:])) == False:  # noqa: E712
            # nd input where some trailing are not one
            msg = "Checking {0} inputs - trailing dims of input '{1}' {2} must be singletons (length=1)"
            print(msg.format(func_name, names[idx], xx.shape))
            raise ValueError(msg)
        elif xx.ndim == 1:
            # Vector input - add a dummy dimension
            msg = "Checking {0} inputs - Adding dummy dimension to input '{1}'"
            print(msg.format(func_name, names[idx]))
            out_args[idx] = out_args[idx][:, np.newaxis]

    if len(out_args) == 1:
        return out_args[0]
    else:
        return out_args

def _nsamples_warn(N, max_imfs):
    if max_imfs is None:
        return
    if N < 2**(max_imfs+1):
        msg = 'Inputs samples ({0}) is small for specified max_imfs ({1})'
        msg += ' very likely that {2} or fewer imfs are returned'
        logger.warning(msg.format(N, max_imfs, np.floor(np.log2(N)).astype(int)-1))


def check_sift_continue(X, residual, layer, max_imfs=None, sift_thresh=1e-8, energy_thresh=50,
                        rilling_thresh=None, envelope_opts=None, extrema_opts=None,
                        merge_tests=True):
    """Run checks to see if siftiing should continue into another layer.

    Parameters
    ----------
    X : ndarray
        1D array containing the data being decomposed
    residual : ndarray
        1D array containing the current residuals (X - imfs so far)
    layer : int
        Current IMF number being decomposed
    max_imf : int
        Largest number of IMFs to compute
    sift_thresh : float
         The threshold at which the overall sifting process will stop.
         (Default value = 1e-8)
    energy_thresh : float
        The difference in energy between the raw data and the residuals in
        decibels at which we stop sifting (default = 50).
    rilling_thresh : tuple or None
        Tuple (or tuple-like) containing three values (sd1, sd2, alpha).
        An evaluation function (E) is defined by dividing the residual by the
        mode amplitude. The sift continues until E < sd1 for the fraction
        (1-alpha) of the data, and E < sd2 for the remainder.
        See section 3.2 of http://perso.ens-lyon.fr/patrick.flandrin/NSIP03.pdf
    envelope_opts : dict or None
        Optional dictionary of keyword options to be passed to emd.interp_envelope
    extrema_opts : dict or None
        Optional dictionary of keyword options to be passed to emd.get_padded_extrema

    Returns
    -------
    bool
        Flag indicating whether to stop sifting.

    """
    continue_sift = [None, None, None, None, None]

    # Check if we've reached the pre-specified number of IMFs
    if max_imfs is not None and layer == max_imfs:
        logger.info('Finishing sift: reached max number of imfs ({0})'.format(layer))
        continue_sift[0] = False
    else:
        continue_sift[0] = True

    # Check if residual has enough peaks to sift again
    pks, _ = _find_extrema(residual)
    trs, _ = _find_extrema(-residual)
    if len(pks) < 2 or len(trs) < 2:
        logger.info('Finishing sift: {0} peaks {1} trough in residual'.format(len(pks), len(trs)))
        continue_sift[1] = False
    else:
        continue_sift[1] = True

    # Optional: Check if the sum-sqr of the resduals is below the sift_thresh
    sumsq_resid = np.abs(residual).sum()
    if sift_thresh is not None and sumsq_resid < sift_thresh:
        logger.info('Finishing sift: reached threshold {0}'.format(sumsq_resid))
        continue_sift[2] = False
    else:
        continue_sift[2] = True

    # Optional: Check if energy_ratio of residual to original signal is below thresh
    energy_ratio = _energy_difference(X, residual)
    if energy_thresh is not None and energy_ratio > energy_thresh:
        logger.info('Finishing sift: reached energy ratio {0}'.format(energy_ratio))
        continue_sift[3] = False
    else:
        continue_sift[3] = True

    # Optional: Check if the residual is already an IMF with Rilling method -
    # only run if we have enough extrema
    if rilling_thresh is not None and continue_sift[1]:
        upper, lower = interp_envelope(residual, mode='both',
                                       **envelope_opts, extrema_opts=extrema_opts)
        rilling_continue_sift, rilling_metric = stop_imf_rilling(upper, lower, niters=-1)
        if rilling_continue_sift is False:
            logger.info('Finishing sift: reached rilling {0}'.format(rilling_metric))
            continue_sift[4] = False
        else:
            continue_sift[4] = True

    if merge_tests:
        # Merge tests that aren't none - return False for any Falses
        return np.any([x == False for x in continue_sift if x is not None]) == False  # noqa: E712
    else:
        return continue_sift


def _energy_difference(imf, residue):
    """Compute energy change in IMF during a sift.

    Parameters
    ----------
    imf : ndarray
        IMF to be evaluated
    residue : ndarray
        Remaining signal after IMF removal

    Returns
    -------
    float
        Energy difference in decibels

    Notes
    -----
    This function is used during emd.sift.stop_imf_energy to implement the
    energy-difference sift-stopping method defined in section 3.2.4 of
    https://doi.org/10.1016/j.ymssp.2007.11.028

    """
    sumsqr = np.sum(imf**2)
    imf_energy = 20 * np.log10(sumsqr, where=sumsqr > 0)
    sumsqr = np.sum(residue ** 2)
    resid_energy = 20 * np.log10(sumsqr, where=sumsqr > 0)
    return imf_energy-resid_energy


def get_next_imf(X, env_step_size=1, max_iters=1000, energy_thresh=50,
                 stop_method='sd', sd_thresh=.1, rilling_thresh=(0.05, 0.5, 0.05),
                 envelope_opts=None, extrema_opts=None):
    """Compute the next IMF from a data set.

    This is a helper function used within the more general sifting functions.

    Parameters
    ----------
    X : ndarray [nsamples x 1]
        1D input array containing the time-series data to be decomposed
    env_step_size : float
        Scaling of envelope prior to removal at each iteration of sift. The
        average of the upper and lower envelope is muliplied by this value
        before being subtracted from the data. Values should be between
        0 > x >= 1 (Default value = 1)
    max_iters : int > 0
        Maximum number of iterations to compute before throwing an error
    energy_thresh : float > 0
        Threshold for energy difference (in decibels) between IMF and residual
        to suggest stopping overall sift. (Default is None, recommended value is 50)
    stop_method : {'sd','rilling','fixed'}
        Flag indicating which metric to use to stop sifting and return an IMF.
    sd_thresh : float
        Used if 'stop_method' is 'sd'. The threshold at which the sift of each
        IMF will be stopped. (Default value = .1)
    rilling_thresh : tuple
        Used if 'stop_method' is 'rilling', needs to contain three values (sd1, sd2, alpha).
        An evaluation function (E) is defined by dividing the residual by the
        mode amplitude. The sift continues until E < sd1 for the fraction
        (1-alpha) of the data, and E < sd2 for the remainder.
        See section 3.2 of http://perso.ens-lyon.fr/patrick.flandrin/NSIP03.pdf

    Returns
    -------
    proto_imf : ndarray
        1D vector containing the next IMF extracted from X
    continue_flag : bool
        Boolean indicating whether the sift can be continued beyond this IMF

    Other Parameters
    ----------------
    envelope_opts : dict
        Optional dictionary of keyword arguments to be passed to emd.interp_envelope
    extrema_opts : dict
        Optional dictionary of keyword options to be passed to emd.get_padded_extrema

    See Also
    --------
    emd.sift.sift
    emd.sift.interp_envelope

    """
    X = ensure_1d_with_singleton([X], ['X'], 'get_next_imf')

    if envelope_opts is None:
        envelope_opts = {}

    proto_imf = X.copy()

    continue_imf = True  # TODO - assess this properly here, return input if already passing!

    continue_flag = True
    niters = 0
    while continue_imf:

        if stop_method != 'fixed':
            if niters == 3*max_iters//4:
                logger.debug('Sift reached {0} iterations, taking a long time to coverge'.format(niters))
            elif niters > max_iters:
                msg = 'Sift failed. No covergence after {0} iterations'.format(niters)
                raise EMDSiftCovergeError(msg)
        niters += 1

        # Compute envelopes, local mean and next proto imf
        upper, lower = interp_envelope(proto_imf, mode='both',
                                       **envelope_opts, extrema_opts=extrema_opts)

        # If upper or lower are None we should stop sifting altogether
        if upper is None or lower is None:
            continue_flag = False
            continue_imf = False
            logger.debug('Finishing sift: IMF has no extrema')
            continue

        # Find local mean
        avg = np.mean([upper, lower], axis=0)[:, None]

        # Remove local mean estimate from proto imf
        #x1 = proto_imf - avg
        next_proto_imf = proto_imf - (env_step_size*avg)

        # Evaluate if we should stop the sift - methods are very different in
        # requirements here...

        # Stop sifting if we pass threshold
        if stop_method == 'sd':
            # Cauchy criterion
            stop, _ = stop_imf_sd(proto_imf, next_proto_imf, sd=sd_thresh, niters=niters)
        elif stop_method == 'rilling':
            # Rilling et al 2003 - this actually evaluates proto_imf NOT next_proto_imf
            stop, _ = stop_imf_rilling(upper, lower, niters=niters,
                                       sd1=rilling_thresh[0],
                                       sd2=rilling_thresh[1],
                                       tol=rilling_thresh[2])
            if stop:
                next_proto_imf = proto_imf
        elif stop_method == 'energy':
            # Rato et al 2008
            # Compare energy of signal at start of sift with energy of envelope average
            stop, _ = stop_imf_energy(X, avg, thresh=energy_thresh, niters=niters)
        elif stop_method == 'fixed':
            stop = stop_imf_fixed_iter(niters, max_iters)
        else:
            raise ValueError("stop_method '{0}' not recogised".format(stop_method))

        proto_imf = next_proto_imf

        if stop:
            continue_imf = False
            continue

    if proto_imf.ndim == 1:
        proto_imf = proto_imf[:, None]

    return proto_imf, continue_flag

def stop_imf_sd(proto_imf, prev_imf, sd=0.2, niters=None):
    """Compute the sd sift stopping metric.

    Parameters
    ----------
    proto_imf : ndarray
        A signal which may be an IMF
    prev_imf : ndarray
        The previously identified IMF
    sd : float
        The stopping threshold
    niters : int
        Number of sift iterations currently completed
    niters : int
        Number of sift iterations currently completed

    Returns
    -------
    bool
        A flag indicating whether to stop siftingg
    float
        The SD metric value

    """
    metric = np.sum((prev_imf - proto_imf)**2) / np.sum(prev_imf**2)

    stop = metric < sd

    if stop:
        logger.verbose('Sift stopped by SD-thresh in {0} iters with sd {1}'.format(niters, metric))
    else:
        logger.debug('SD-thresh stop metric evaluated at iter {0} is : {1}'.format(niters, metric))

    return stop, metric


    
    
    
    
    



