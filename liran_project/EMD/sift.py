import torch
import torch.nn.functional as F
from scipy.signal import argrelextrema, peak_prominences
import math


def torch_get_next_imf(X, env_step_size=1, max_iters=1000, energy_thresh=50,
                        stop_method='sd', sd_thresh=.1, rilling_thresh=(0.05, 0.5, 0.05),
                        envelope_opts=None, extrema_opts=None):
    """Compute the next IMF from a data set using PyTorch.

    Parameters
    ----------
    X : torch.Tensor [nsamples x 1]
        1D input tensor containing the time-series data to be decomposed
    env_step_size : float
        Scaling of envelope prior to removal at each iteration of sift.
    max_iters : int > 0
        Maximum number of iterations to compute before throwing an error
    energy_thresh : float > 0
        Threshold for energy difference (in decibels) between IMF and residual
    stop_method : {'sd','rilling','fixed'}
        Flag indicating which metric to use to stop sifting and return an IMF.
    sd_thresh : float
        Used if 'stop_method' is 'sd'. The threshold at which the sift of each
        IMF will be stopped.
    rilling_thresh : tuple
        Used if 'stop_method' is 'rilling', needs to contain three values (sd1, sd2, alpha).

    Returns
    -------
    proto_imf : torch.Tensor
        1D tensor containing the next IMF extracted from X
    continue_flag : bool
        Boolean indicating whether the sift can be continued beyond this IMF
    """
    X = ensure_1d_with_singleton([X], ['X'], 'torch_get_next_imf')

    if envelope_opts is None:
        envelope_opts = {}

    proto_imf = X.clone()

    continue_imf = True
    continue_flag = True
    niters = 0
    while continue_imf:
        if stop_method != 'fixed':
            if niters == 3 * max_iters // 4:
                print(f'Sift reached {niters} iterations, taking a long time to converge')
            elif niters > max_iters*4:
                raise RuntimeError(f'Sift failed. No convergence after {niters} iterations')

        niters += 1

        # Compute envelopes, local mean and next proto imf
        upper, lower = torch_interp_envelope(proto_imf, mode='both',
                                             **envelope_opts, extrema_opts=extrema_opts)

        if upper is None or lower is None:
            continue_flag = False
            continue_imf = False
            print('Finishing sift: IMF has no extrema')
            continue

        # Find local mean
        avg = (upper + lower) / 2

        # Remove local mean estimate from proto imf
        next_proto_imf = proto_imf - (env_step_size * avg)

        # Evaluate if we should stop the sift
        if stop_method == 'sd':
            stop, _ = torch_stop_imf_sd(proto_imf, next_proto_imf, sd=sd_thresh, niters=niters)
        elif stop_method == 'rilling':
            stop, _ = torch_stop_imf_rilling(upper, lower, niters=niters,
                                               sd1=rilling_thresh[0],
                                               sd2=rilling_thresh[1],
                                               tol=rilling_thresh[2])
            if stop:
                next_proto_imf = proto_imf
        elif stop_method == 'energy':
            stop, _ = torch_stop_imf_energy(X, avg, thresh=energy_thresh, niters=niters)
        elif stop_method == 'fixed':
            stop = torch_stop_imf_fixed_iter(niters, max_iters)
        else:
            raise ValueError(f"stop_method '{stop_method}' not recognized")

        proto_imf = next_proto_imf

        if stop:
            continue_imf = False
            continue

    if proto_imf.ndim == 1:
        proto_imf = proto_imf.unsqueeze(1)

    return proto_imf, continue_flag


def torch_energy_difference(imf, residue):
    """Compute energy change in IMF during a sift using PyTorch.

    Parameters
    ----------
    imf : torch.Tensor
        IMF to be evaluated
    residue : torch.Tensor
        Remaining signal after IMF removal

    Returns
    -------
    float
        Energy difference in decibels
    """
    sumsqr = torch.sum(imf ** 2)
    imf_energy = 20 * torch.log10(sumsqr + 1e-10)  # add small value to avoid log(0)
    sumsqr = torch.sum(residue ** 2)
    resid_energy = 20 * torch.log10(sumsqr + 1e-10)
    return imf_energy - resid_energy


def torch_stop_imf_energy(imf, residue, thresh=50, niters=None):
    """Compute energy change in IMF during a sift using PyTorch.

    Parameters
    ----------
    imf : torch.Tensor
        IMF to be evaluated
    residue : torch.Tensor
        Average of the upper and lower envelopes
    thresh : float
        Energy ratio threshold (default=50)
    niters : int
        Number of sift iterations currently completed

    Returns
    -------
    bool
        A flag indicating whether to stop sifting
    float
        Energy difference in decibels
    """
    diff = torch_energy_difference(imf, residue)
    stop = diff > thresh

    if stop:
        print(f'Sift stopped by Energy Ratio in {niters} iters with difference of {diff}dB')
    else:
        print(f'Energy Ratio evaluated at iter {niters} is: {diff}dB')

    return stop, diff


def torch_stop_imf_sd(proto_imf, prev_imf, sd=0.2, niters=None):
    """Compute the sd sift stopping metric using PyTorch.

    Parameters
    ----------
    proto_imf : torch.Tensor
        A signal which may be an IMF
    prev_imf : torch.Tensor
        The previously identified IMF
    sd : float
        The stopping threshold
    niters : int
        Number of sift iterations currently completed

    Returns
    -------
    bool
        A flag indicating whether to stop sifting
    float
        The SD metric value
    """
    metric = torch.sum((prev_imf - proto_imf) ** 2) / torch.sum(prev_imf ** 2)

    stop = metric < sd

    if stop:
        print(f'Sift stopped by SD-thresh in {niters} iters with sd {metric}')
    else:
        print(f'SD-thresh stop metric evaluated at iter {niters} is: {metric}')

    return stop, metric


def torch_stop_imf_rilling(upper_env, lower_env, sd1=0.05, sd2=0.5, tol=0.05, niters=None):
    """Compute the Rilling et al 2003 sift stopping metric using PyTorch.

    Parameters
    ----------
    upper_env : torch.Tensor
        The upper envelope of a proto-IMF
    lower_env : torch.Tensor
        The lower envelope of a proto-IMF
    sd1 : float
        The maximum threshold for globally small differences from zero-mean
    sd2 : float
        The maximum threshold for locally large differences from zero-mean
    tol : float (0 < tol < 1)
        (1-tol) defines the proportion of time which may contain large deviations
    niters : int
        Number of sift iterations currently completed

    Returns
    -------
    bool
        A flag indicating whether to stop sifting
    float
        The SD metric value
    """
    avg_env = (upper_env + lower_env) / 2
    amp = torch.abs(upper_env - lower_env) / 2

    eval_metric = torch.abs(avg_env) / (amp + 1e-10)  # prevent division by zero

    metric = torch.mean(eval_metric > sd1)
    continue1 = metric > tol
    continue2 = torch.any(eval_metric > sd2)

    stop = (continue1 or continue2) == False

    if stop:
        print(f'Sift stopped by Rilling-metric in {niters} iters (val={metric})')
    else:
        print(f'Rilling stop metric evaluated at iter {niters} is: {metric}')

    return stop, metric


def torch_stop_imf_fixed_iter(niters, max_iters):
    """Compute the fixed-iteration sift stopping metric using PyTorch.

    Parameters
    ----------
    niters : int
        Number of sift iterations currently completed
    max_iters : int
        Maximum number of sift iterations to be completed

    Returns
    -------
    bool
        A flag indicating whether to stop sifting
    """
    stop = niters == max_iters

    if stop:
        print(f'Sift stopped at fixed number of {niters} iterations')

    return stop


def torch_nsamples_warn(N, max_imfs):
    """Warn if the number of samples is too small for the specified max_imfs."""
    if max_imfs is None:
        return
    if N < 2 ** (max_imfs + 1):
        msg = 'Inputs samples ({0}) is small for specified max_imfs ({1})'
        msg += ' very likely that {2} or fewer imfs are returned'
        print(msg.format(N, max_imfs, max_imfs))


def ensure_1d_with_singleton(inputs, names, func_name):
    """Ensure input is a 1D tensor and has a singleton dimension."""
    for i, (input_tensor, name) in enumerate(zip(inputs, names)):
        input_tensor = torch.tensor(input_tensor).squeeze()
        if input_tensor.ndim != 1:
            raise ValueError(f"{func_name} expects {name} to be 1D but got {input_tensor.ndim}D")
        if input_tensor.shape[0] == 0:
            raise ValueError(f"{func_name} expects {name} to be non-empty")
        # Ensuring singleton dimension
        inputs[i] = input_tensor.view(-1, 1)
    return inputs[0]


def torch_interp_envelope(X, mode='both', interp_method='linear', extrema_opts=None,
                    ret_extrema=False, trim=True):
    """Interpolate the amplitude envelope of a signal.

    Parameters
    ----------
    X : torch.Tensor
        Input signal
    mode : {'upper', 'lower', 'combined'}
         Flag to set which envelope should be computed (Default value = 'upper')
    interp_method : {'linear'}
         Flag to indicate which interpolation method should be used (Default value = 'linear')

    Returns
    -------
    torch.Tensor
        Interpolated amplitude envelope
    """
    if not extrema_opts:
        extrema_opts = {'pad_width': 2, 'loc_pad_opts': None, 'mag_pad_opts': None}
    else:
        extrema_opts = extrema_opts.copy()

    if interp_method not in ['linear']:
        raise ValueError("Invalid interp_method value")

    if mode == 'upper':
        extr = torch_get_padded_extrema(X, mode='peaks', **extrema_opts)
    elif mode == 'lower':
        extr = torch_get_padded_extrema(X, mode='troughs', **extrema_opts)
    elif mode == 'both':
        extr = torch_get_padded_extrema(X, mode='both', **extrema_opts)
    elif mode == 'combined':
        extr = torch_get_padded_extrema(X, mode='abs_peaks', **extrema_opts)
    else:
        raise ValueError("Mode not recognised. Use mode= 'upper'|'lower'|'combined'")

    if extr[0] is None:
        if mode == 'both':
            return None, None
        else:
            return None

    if mode == 'both':
        lower = torch_run_scipy_interp(extr[0], extr[1], lenx=X.size(0), trim=trim, interp_method=interp_method)
        upper = torch_run_scipy_interp(extr[2], extr[3], lenx=X.size(0), trim=trim, interp_method=interp_method)
        env = (upper, lower)
    else:
        env = torch_run_scipy_interp(extr[0], extr[1], lenx=X.size(0), interp_method=interp_method, trim=trim)

    if ret_extrema:
        return env, extr
    else:
        return env
    
def torch_run_interp(locs, pks, lenx, interp_method='linear', trim=True):
    """
    Perform interpolation on envelope using PyTorch.

    Parameters
    ----------
    locs : torch.Tensor
        Locations of the extrema (peaks or troughs).
    pks : torch.Tensor
        Values of the extrema at the corresponding locations.
    lenx : int
        Length of the input signal for trimming.
    interp_method : {'linear'}
        Interpolation method. Default is 'linear'.
    trim : bool
        Whether to trim the output to match the original signal length.

    Returns
    -------
    torch.Tensor
        Interpolated envelope.
    """

    # Interpolate envelope using linear interpolation (other methods can be added as needed)
    t = torch.arange(locs[0], locs[-1] + 1)
    
    if interp_method == 'linear':
        env = torch.interp(t, locs, pks)
    else:
        raise ValueError(f"Interpolation method '{interp_method}' is not implemented.")

    if trim:
        # Ensure that the envelope is within the input signal's length
        t_max = torch.arange(locs[0], locs[-1] + 1)
        tinds = (t_max >= 0) & (t_max < lenx)
        env = env[tinds]

        if env.shape[0] != lenx:
            msg = f"Envelope length ({env.shape[0]}) does not match input data length ({lenx})."
            raise ValueError(msg)

    return env

def torch_sift(X, sift_thresh=1e-8, energy_thresh=50, rilling_thresh=None,
                max_imfs=None, verbose=None, return_residual=True,
                imf_opts=None, envelope_opts=None, extrema_opts=None):
    """Compute Intrinsic Mode Functions from an input data vector.

    This function implements the original sift algorithm [1]_.

    Parameters
    ----------
    X : torch.Tensor
        1D input tensor containing the time-series data to be decomposed
    sift_thresh : float
         The threshold at which the overall sifting process will stop. (Default value = 1e-8)
    max_imfs : int
         The maximum number of IMFs to compute. (Default value = None)

    Returns
    -------
    imf: torch.Tensor
        2D tensor [samples x nimfs] containing the Intrinsic Mode Functions from the decomposition of X.

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
    left to default.

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
    rilling_thresh = torch_set_rilling_defaults(rilling_thresh)

    X = ensure_1d_with_singleton([X], ['X'], 'torch_sift')

    _nsamples_warn(X.shape[0], max_imfs)

    layer = 0
    continue_sift = torch_check_sift_continue(X, X, layer,
                                               max_imfs=max_imfs,
                                               sift_thresh=None,
                                               energy_thresh=None,
                                               rilling_thresh=rilling_thresh,
                                               envelope_opts=envelope_opts,
                                               extrema_opts=extrema_opts,
                                               merge_tests=True)

    proto_imf = X.clone()

    while continue_sift:

        next_imf, continue_sift = torch_get_next_imf(proto_imf,
                                                       envelope_opts=envelope_opts,
                                                       extrema_opts=extrema_opts,
                                                       **imf_opts)

        if layer == 0:
            imf = next_imf
        else:
            imf = torch.cat((imf, next_imf), dim=1)

        proto_imf = X - imf.sum(dim=1, keepdim=True)
        layer += 1

        continue_sift = torch_check_sift_continue(X, proto_imf, layer,
                                                    max_imfs=max_imfs,
                                                    sift_thresh=sift_thresh,
                                                    energy_thresh=energy_thresh,
                                                    rilling_thresh=rilling_thresh,
                                                    envelope_opts=envelope_opts,
                                                    extrema_opts=extrema_opts,
                                                    merge_tests=True)

    if torch.sum(torch.abs(proto_imf)) != 0:
        imf = torch.cat((imf, proto_imf.unsqueeze(1)), dim=1)

    return imf


def torch_check_sift_continue(X, residual, layer, max_imfs=None, sift_thresh=1e-8, energy_thresh=50,
                               rilling_thresh=None, envelope_opts=None, extrema_opts=None,
                               merge_tests=True):
    """Run checks to see if sifting should continue into another layer.

    Parameters
    ----------
    X : torch.Tensor
        1D tensor containing the data being decomposed
    residual : torch.Tensor
        1D tensor containing the current residuals (X - imfs so far)
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

    Returns
    -------
    bool
        Flag indicating whether to stop sifting.

    """
    continue_sift = [None, None, None, None, None]

    # Check if we've reached the pre-specified number of IMFs
    if max_imfs is not None and layer == max_imfs:
        print('Finishing sift: reached max number of imfs ({0})'.format(layer))
        continue_sift[0] = False
    else:
        continue_sift[0] = True

    # Check if residual has enough peaks to sift again
    pks, _ = torch_find_extrema(residual)
    trs, _ = torch_find_extrema(-residual)
    if len(pks) < 2 or len(trs) < 2:
        continue_sift[1] = False
    else:
        continue_sift[1] = True

    # Optional: Check if the sum-sqr of the residuals is below the sift_thresh
    sumsq_resid = torch.abs(residual).sum()
    if sift_thresh is not None and sumsq_resid < sift_thresh:
        print('Finishing sift: reached threshold {0}'.format(sumsq_resid))
        continue_sift[2] = False
    else:
        continue_sift[2] = True

    # Optional: Check if energy_ratio of residual to original signal is below thresh
    energy_ratio = torch_energy_difference(X, residual)
    if energy_thresh is not None and energy_ratio > energy_thresh:
        continue_sift[3] = False
    else:
        continue_sift[3] = True

    # Optional: Check if the residual is already an IMF with Rilling method -
    # only run if we have enough extrema
    if rilling_thresh is not None and continue_sift[1]:
        upper, lower = torch_interp_envelope(residual, mode='both',
                                             **envelope_opts, extrema_opts=extrema_opts)
        rilling_continue_sift, rilling_metric = torch_stop_imf_rilling(upper, lower, niters=-1)
        if rilling_continue_sift is False:
            continue_sift[4] = False
        else:
            continue_sift[4] = True

    if merge_tests:
        return not any(x is False for x in continue_sift if x is not None)
    else:
        return continue_sift
    

def torch_get_padded_extrema(X, pad_width=2, mode='peaks', parabolic_extrema=False,
                              loc_pad_opts=None, mag_pad_opts=None, method='rilling'):
    """Identify and pad the extrema in a signal.

    This function returns a set of extrema from a signal including padded
    extrema at the edges of the signal. Padding is carried out using torch.

    Parameters
    ----------
    X : torch.Tensor
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
        Optional dictionary of options to be passed to torch.nn.functional.pad when padding extrema locations
    mag_pad_opts : dict
        Optional dictionary of options to be passed to torch.nn.functional.pad when padding extrema magnitudes

    Returns
    -------
    locs : torch.Tensor
        Location of extrema in samples
    mags : torch.Tensor
        Magnitude of each extrema
    """
    if (mode == 'abs_peaks') and (method == 'rilling'):
        msg = "torch_get_padded_extrema mode 'abs_peaks' is incompatible with method 'rilling'"
        raise ValueError(msg)

    if X.ndim == 2:
        X = X[:, 0]

    if mode == 'both' or method == 'rilling':
        max_locs, max_ext = _torch_find_extrema(X, parabolic_extrema=parabolic_extrema)
        min_locs, min_ext = _torch_find_extrema(-X, parabolic_extrema=parabolic_extrema)
        min_ext = -min_ext
        print(f'found {len(min_locs)} minima and {len(max_locs)} maxima on mode {mode}')
    elif mode == 'peaks':
        max_locs, max_ext = _torch_find_extrema(X, parabolic_extrema=parabolic_extrema)
        print(f'found {len(max_locs)} maxima on mode {mode}')
    elif mode == 'troughs':
        max_locs, max_ext = _torch_find_extrema(-X, parabolic_extrema=parabolic_extrema)
        max_ext = -max_ext
        print(f'found {len(max_locs)} minima on mode {mode}')
    elif mode == 'abs_peaks':
        max_locs, max_ext = _torch_find_extrema(X.abs(), parabolic_extrema=parabolic_extrema)
        print(f'found {len(max_locs)} extrema on mode {mode}')
    else:
        raise ValueError(f'Mode {mode} not recognized by torch_get_padded_extrema')

    # Return nothing if we don't have enough extrema
    if (len(max_locs) == 0) or (max_locs.size <= 1):
        print('Not enough extrema to pad.')
        return None, None
    elif (mode == 'both' or method == 'rilling') and len(min_locs) <= 1:
        print('Not enough extrema to pad 2.')
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
        ret = _torch_pad_extrema_numpy(max_locs, max_ext,
                                       X.shape[0], pad_width,
                                       loc_pad_opts, mag_pad_opts)
        if mode == 'both':
            ret2 = _torch_pad_extrema_numpy(min_locs, min_ext,
                                            X.shape[0], pad_width,
                                            loc_pad_opts, mag_pad_opts)
            ret = (ret2[0], ret2[1], ret[0], ret[1])
    elif method == 'rilling':
        ret = _torch_pad_extrema_rilling(min_locs, max_locs, X, pad_width)
        # Inefficient to use rilling for just peaks or troughs, but handle it
        # just in case.
        if mode == 'peaks':
            ret = ret[2:]
        elif mode == 'troughs':
            ret = ret[:2]

    return ret


def _torch_pad_extrema_numpy(locs, mags, lenx, pad_width, loc_pad_opts, mag_pad_opts):
    """Pad extrema using a direct call to torch.nn.functional.pad.

    Extra paddings are carried out if the padded values do not span the whole
    range of the original time-series (defined by lenx)

    Parameters
    ----------
    locs : torch.Tensor
        Location of extrema in time
    mags : torch.Tensor
        Magnitude of each extrema
    lenx : int
        Length of the time-series from which locs and mags were identified
    pad_width : int
        Number of extra extrema to pad
    loc_pad_opts : dict
        Dictionary of arguments passed to torch.nn.functional.pad to generate new extrema locations
    mag_pad_opts : dict
        Dictionary of arguments passed to torch.nn.functional.pad to generate new extrema magnitudes

    Returns
    -------
    torch.Tensor
        Location of all extrema (including padded and original points) in time
    torch.Tensor
        Magnitude of each extrema (including padded and original points)
    """
    print(f"Padding {pad_width} extrema in signal X {lenx} using method 'numpypad'")

    if not loc_pad_opts:  # Empty dict evaluates to False
        loc_pad_opts = {'mode': 'reflect'}  # Default padding mode
    else:
        loc_pad_opts = loc_pad_opts.copy()  # Don't work in place...

    if not mag_pad_opts:  # Empty dict evaluates to False
        mag_pad_opts = {'mode': 'median'}  # Default padding mode
    else:
        mag_pad_opts = mag_pad_opts.copy()  # Don't work in place...

    # Determine how much padding to use
    if locs.size < pad_width:
        pad_width = locs.size

    # Return now if we're not padding
    if (pad_width is None) or (pad_width == 0):
        return locs, mags

    # Pad peak locations
    ret_locs = F.pad(locs, (pad_width, pad_width), mode='replicate')

    # Pad peak magnitudes
    ret_mag = F.pad(mags, (pad_width, pad_width), mode='replicate')

    # Keep padding if the locations don't stretch to the edge
    count = 0
    while (ret_locs.max() < lenx) or (ret_locs.min() >= 0):
        print(f'Padding again - first ext {ret_locs.min()}, last ext {ret_locs.max()}')
        print(ret_locs)
        ret_locs = F.pad(ret_locs, (pad_width, pad_width), mode='replicate')
        ret_mag = F.pad(ret_mag, (pad_width, pad_width), mode='replicate')
        count += 1

    return ret_locs, ret_mag


def _torch_pad_extrema_rilling(indmin, indmax, X, pad_width):
    """Pad extrema using the method from Rilling.

    This is based on original matlab code in boundary_conditions_emd.m

    Unlike the numpypad method - this approach pads both the maxima and minima of the signal together.

    Parameters
    ----------
    indmin : torch.Tensor
        Location of minima in time
    indmax : torch.Tensor
        Location of maxima in time
    X : torch.Tensor
        Original time-series
    pad_width : int
        Number of extra extrema to pad

    Returns
    -------
    tmin : torch.Tensor
        Location of all minima (including padded and original points) in time
    xmin : torch.Tensor
        Magnitude of each minima (including padded and original points)
    tmax : torch.Tensor
        Location of all maxima (including padded and original points) in time
    xmax : torch.Tensor
        Magnitude of each maxima (including padded and original points)
    """
    print(f"Padding {pad_width} extrema in signal X {X.shape} using method 'rilling'")

    t = torch.arange(len(X))

    # Pad START
    if indmax[0] < indmin[0]:
        if X[0] > X[indmin[0]]:
            lmax = indmax[1:pad_width+1].flip(0)
            lmin = indmin[:pad_width].flip(0)
            lsym = indmax[0]
        else:
            lmax = indmax[:pad_width].flip(0)
            lmin = torch.cat((indmin[:pad_width-1].flip(0), torch.tensor([0], dtype=torch.long)))
            lsym = 0
    else:
        lmax = torch.cat((indmax[:pad_width-1].flip(0), torch.tensor([0], dtype=torch.long)))
        lmin = indmin[1:pad_width+1].flip(0)
        lsym = indmin[0]

    # Pad END
    if indmax[-1] > indmin[-1]:
        if X[-1] > X[indmin[-1]]:
            rmax = indmax[-pad_width-1:-1].flip(0)
            rmin = indmin[-pad_width:].flip(0)
            rsym = indmax[-1]
        else:
            rmax = indmax[-pad_width:].flip(0)
            rmin = torch.cat((indmin[-pad_width+1:-1].flip(0), torch.tensor([len(X)-1], dtype=torch.long)))
            rsym = len(X) - 1
    else:
        rmax = torch.cat((indmax[-pad_width+1:-1].flip(0), torch.tensor([len(X)-1], dtype=torch.long)))
        rmin = indmin[-pad_width-1:-1].flip(0)
        rsym = indmin[-1]

    # Build the outputs
    tmin = torch.cat((lmin, indmin, rmin))
    xmin = torch.cat((X[lmax], X[indmin], X[rmin]))

    tmax = torch.cat((lmax, indmax, rmax))
    xmax = torch.cat((X[lmax], X[indmax], X[rmax]))

    return tmin, xmin, tmax, xmax


def _torch_find_extrema(X, parabolic_extrema=False):
    """Find local minima and maxima of a signal.

    This function uses scipy's argrelextrema to find local minima and maxima.

    Parameters
    ----------
    X : torch.Tensor
        Input signal
    parabolic_extrema : bool
        Flag indicating whether to refine extrema positions by parabolic interpolation

    Returns
    -------
    locs : torch.Tensor
        Location of extrema in samples
    exts : torch.Tensor
        Magnitude of each extrema
    """
    # Get local maxima
    locs = argrelextrema(X.cpu().numpy(), torch.tensor).flatten()
    exts = X[locs]

    if parabolic_extrema:
        # Refine locations using parabolic interpolation
        locs = locs + (exts[1:] - exts[:-1]) / (2 * (exts[1:] - 2 * exts + exts[:-1])) 

    return locs, exts


def torch_ensure_1d_with_singleton(to_check, names, func_name):
    """Check that a set of tensors are all vectors with singleton second dimensions.

    1D tensors will have a singleton second dimension added, and an error will be
    raised for non-singleton 2D or greater than 2D inputs.

    Parameters
    ----------
    to_check : list of torch.Tensor
        List of tensors to check for equal dimensions
    names : list
        List of variable names for tensors in to_check
    func_name : str
        Name of the function calling ensure_equal_dims

    Returns
    -------
    out : torch.Tensor or list
        Copy of tensors in to_check with '1D with singleton' shape.

    Raises
    ------
    ValueError
        If any input is a 2D or greater tensor
    """
    out_args = list(to_check)
    
    for idx, xx in enumerate(to_check):
        if (xx.ndim >= 2) and torch.all(xx.shape[1:] == 1):
            # ND input where all trailing dimensions are ones
            msg = "Checking {0} inputs - Trimming trailing singletons from input '{1}' (input size {2})"
            print(msg.format(func_name, names[idx], xx.shape))  # Replace logger.debug with print
            out_args[idx] = xx.squeeze()[:, None]  # Add singleton dimension
        elif (xx.ndim >= 2) and not torch.all(xx.shape[1:] == 1):  # noqa: E712
            # ND input where some trailing are not one
            msg = "Checking {0} inputs - trailing dims of input '{1}' {2} must be singletons (length=1)"
            print(msg.format(func_name, names[idx], xx.shape))  # Replace logger.error with print
            raise ValueError(msg)
        elif xx.ndim == 1:
            # Vector input - add a dummy dimension
            msg = "Checking {0} inputs - Adding dummy dimension to input '{1}'"
            print(msg.format(func_name, names[idx]))  # Replace logger.debug with print
            out_args[idx] = out_args[idx][:, None]  # Add singleton dimension

    if len(out_args) == 1:
        return out_args[0]
    else:
        return out_args
    

def torch_set_rilling_defaults(rilling_thresh):
    rilling_thresh = (0.05, 0.5, 0.05) if rilling_thresh is True else rilling_thresh
    return rilling_thresh


def _nsamples_warn(N, max_imfs):
    if max_imfs is None:
        return
    if N < 2**(max_imfs+1):
        msg = 'Inputs samples ({0}) is small for specified max_imfs ({1})'
        msg += ' very likely that {2} or fewer imfs are returned'
        print(msg.format(N, max_imfs, max_imfs))
        
        
def torch_find_extrema(X, peak_prom_thresh=None, parabolic_extrema=False):
    """Identify extrema in the signal and optionally refine using parabolic interpolation."""
    
    def torch_argrelextrema(data, comparator):
        """
        Find relative extrema (similar to scipy.signal.argrelextrema)
        Using simple comparison and padding the tensor.
        """
        data = data.squeeze()
        if data.dim() != 1:
            raise ValueError("Input data must be a 1D tensor")

        # Compare data points with their neighbors
        greater_than_next = comparator(data[1:], data[:-1])
        greater_than_prev = comparator(data[:-1], data[1:])
        
        # Find locations where both conditions are true
        ext_locs = torch.nonzero(greater_than_next[:-1] & greater_than_prev[1:]) + 1  # Shift by one due to indexing
        
        return ext_locs.squeeze(1)
    
    ext_locs = torch_argrelextrema(X, torch.gt)

    if ext_locs.numel() == 0:
        return torch.tensor([]), torch.tensor([])

    if peak_prom_thresh is not None:
        prom = torch_peak_prominences(X, ext_locs)  # This needs a custom torch_peak_prominences function
        keeps = prom > peak_prom_thresh
        ext_locs = ext_locs[keeps]

    if parabolic_extrema:
        y = torch.stack([X[ext_locs - 1], X[ext_locs], X[ext_locs + 1]], dim=0)
        ext_locs, max_pks = torch_compute_parabolic_extrema(y, ext_locs)
        return ext_locs, max_pks
    else:
        return ext_locs, X[ext_locs]
    

import torch

def torch_compute_parabolic_extrema(y, locs):
    """Compute a parabolic refinement of extrema locations.

    Parabolic refinement is computed from triplets of points based on the
    method described in section 3.2.1 from Rato 2008.

    Parameters
    ----------
    y : torch.Tensor
        A [3 x nextrema] tensor containing the points immediately around the
        extrema in a time-series.
    locs : torch.Tensor
        A [nextrema] length tensor containing x-axis positions of the extrema.

    Returns
    -------
    torch.Tensor
        The estimated y-axis values of the interpolated extrema.
    torch.Tensor
        The estimated x-axis values of the interpolated extrema.
    """
    # Define the inverse matrix for the parabola equation
    w_inv = torch.tensor([[.5, -1, .5], [-5/2, 4, -3/2], [3, -3, 1]], dtype=torch.float32)
    
    # Compute a, b, c coefficients from y using matrix multiplication
    abc = torch.matmul(w_inv, y)

    # Find coordinates of extrema from parameters abc
    tp = -abc[1, :] / (2 * abc[0, :])
    t = tp - 2 + locs
    y_hat = tp * abc[0, :]**2 + abc[1, :] * tp + abc[2, :]

    return t, y_hat


def torch_peak_prominences(x, peaks, wlen=None):
    """
    Calculate the prominence of each peak in a signal.

    Parameters
    ----------
    x : torch.Tensor
        A signal with peaks.
    peaks : torch.Tensor
        Indices of peaks in `x`.
    wlen : int, optional
        A window length in samples that optionally limits the evaluated area for
        each peak to a subset of `x`.

    Returns
    -------
    prominences : torch.Tensor
        The calculated prominences for each peak in `peaks`.
    left_bases, right_bases : torch.Tensor
        The peaks' bases as indices in `x` to the left and right of each peak.
    """
    
    def _find_bases(x, peaks, direction):
        """Find bases for the prominence calculation (either left or right)."""
        bases = []
        for peak in peaks:
            i = peak
            base = i
            while 0 <= i < x.size(0):
                if direction == 'left':
                    i -= 1
                else:
                    i += 1
                if i < 0 or i >= x.size(0):
                    break
                if x[i] < x[base]:
                    base = i
            bases.append(base)
        return torch.tensor(bases, dtype=torch.long)

    if wlen is not None:
        raise NotImplementedError("Window length functionality is not yet implemented in this version")

    # Find left and right bases for each peak
    left_bases = _find_bases(x, peaks, direction='left')
    right_bases = _find_bases(x, peaks, direction='right')

    # Compute prominences as the difference between peak value and highest base
    prominences = x[peaks] - torch.maximum(x[left_bases], x[right_bases])

    return prominences, left_bases, right_bases
    
            
        
        
        
        
        
        
        