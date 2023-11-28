# WFDB - Physiological Signal Processing

This Python package facilitates the processing and analysis of physiological signals in the WFDB format (WaveForm DataBase). It offers a range of functions for reading records, processing signals, and comparing annotations.

## Installation

```bash
pip install wfdb
```

## Reading Record Headers

- **`wfdb.rdheader()`**
  - Reads a WFDB header file and returns a `Record` or `MultiRecord` object with record descriptors as attributes.
  - Example:
    ```python
    ecg_record = wfdb.rdheader('100', pn_dir='mitdb')
    ```

## Reading and Processing Records

- **`wfdb.rdrecord()`**
  - Reads a WFDB record, returning signal and record descriptors as attributes in a `Record` or `MultiRecord` object.
  - Example:
    ```python
    record = wfdb.rdrecord('sample-data/test01_00s', sampfrom=800, channels=[1, 3])
    ```

## Signal Processing - Heart Rate

- **`wfdb.processing.ann2rr()`**
  - Obtains RR interval series from ECG annotation files.
  - Example:
    ```python
    rr_intervals = wfdb.processing.ann2rr('100', 'atr')
    ```

- **`wfdb.processing.calc_mean_hr()`**
  - Computes mean heart rate in beats per minute from a set of R-R intervals.
  - Example:
    ```python
    mean_heart_rate = wfdb.processing.calc_mean_hr(rr_intervals)
    ```

## Plotting

- **`wfdb.plot.plot_items()`**
  - Subplots individual channels of signals and/or annotations.
  - Example:
    ```python
    wfdb.plot.plot_items(signal=record.p_signal, ann_samp=[ann.sample, ann.sample], title='MIT-BIH Record 100', time_units='seconds', figsize=(10, 4), ecg_grids='all')
    ```

## Annotation Comparison

- **`wfdb.processing.Comparitor()`**
  - A class to implement and hold comparisons between two sets of annotations.
  - Example:
    ```python
    comparitor = wfdb.processing.Comparitor(ann_ref.sample[1:], xqrs.qrs_inds, int(0.1 * fields['fs']), sig[:,0])
    comparitor.compare()
    comparitor.print_summary()
    comparitor.plot()
    ```

For detailed documentation and additional functionalities, refer to the [WFDB documentation](https://wfdb.readthedocs.io/).
```