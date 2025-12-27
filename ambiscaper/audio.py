# CREATED: 4/23/17 15:37 by Justin Salamon <justin.salamon@nyu.edu>

'''
Utility functions for audio processing using FFMPEG (beyond sox). Based on:
https://github.com/mathos/neg23/
'''
import numpy as np
import pyloudnorm
import subprocess
from .ambiscaper_exceptions import AmbiScaperError


def r128stats(filepath):
    """ takes a path to an audio file, returns a dict with the loudness
    stats computed by the ffmpeg ebur128 filter """
    ffargs = ['ffmpeg',
              '-nostats',
              '-i',
              filepath,
              '-filter_complex',
              'ebur128',
              '-f',
              'null',
              '-']
    try:
        proc = subprocess.Popen(ffargs, stderr=subprocess.PIPE,
                                universal_newlines=True)
        stats = proc.communicate()[1]
        summary_index = stats.rfind('Summary:')

        if summary_index == -1:
            raise AmbiScaperError(
                'Unable to find LUFS summary, stats string:\n{:s}'.format(
                    stats))

        summary_list = stats[summary_index:].split()
        i_lufs = float(summary_list[summary_list.index('I:') + 1])
        i_thresh = float(summary_list[summary_list.index('I:') + 4])
        lra = float(summary_list[summary_list.index('LRA:') + 1])
        lra_thresh = float(summary_list[summary_list.index('LRA:') + 4])
        lra_low = float(summary_list[summary_list.index('low:') + 1])
        lra_high = float(summary_list[summary_list.index('high:') + 1])
        stats_dict = {'I': i_lufs, 'I Threshold': i_thresh, 'LRA': lra,
                      'LRA Threshold': lra_thresh, 'LRA Low': lra_low,
                      'LRA High': lra_high}
    except Exception as e:
        raise AmbiScaperError(
            'Unable to obtain LUFS data for {:s}, error message:\n{:s}'.format(
                filepath, e.__str__()))

    return stats_dict


def get_integrated_lufs_old(filepath):
    '''Returns the integrated lufs for an audiofile'''

    loudness_stats = r128stats(filepath)
    return loudness_stats['I']

def get_integrated_lufs(audio_array, samplerate, min_duration=0.5,
                        filter_class='K-weighting', block_size=0.400):
    """
    Returns the integrated LUFS for a numpy array containing
    audio samples.

    For files shorter than 400 ms pyloudnorm throws an error. To avoid this, 
    files shorter than min_duration (by default 500 ms) are self-concatenated 
    until min_duration is reached and the LUFS value is computed for the 
    concatenated file.

    Parameters
    ----------
    audio_array : np.ndarray
        numpy array containing samples or path to audio file for computing LUFS
    samplerate : int
        Sample rate of audio, for computing duration
    min_duration : float
        Minimum required duration for computing LUFS value. Files shorter than
        this are self-concatenated until their duration reaches this value
        for the purpose of computing the integrated LUFS. Caution: if you set
        min_duration < 0.4, a constant LUFS value of -70.0 will be returned for
        all files shorter than 400 ms.
    filter_class : str
        Class of weighting filter used.
        - 'K-weighting' (default)
        - 'Fenton/Lee 1'
        - 'Fenton/Lee 2'
        - 'Dash et al.'
    block_size : float 
        Gating block size in seconds. Defaults to 0.400.
    
    Returns
    -------
    loudness
        Loudness in terms of LUFS 
    """
    duration = audio_array.shape[0] / float(samplerate)
    if duration < min_duration:
        ntiles = int(np.ceil(min_duration / duration))
        audio_array = np.tile(audio_array, (ntiles, 1))
    meter = pyloudnorm.Meter(
        samplerate, filter_class=filter_class, block_size=block_size
    )
    loudness = meter.integrated_loudness(audio_array)
    # silent audio gives -inf, so need to put a lower bound.
    loudness = max(loudness, -70) 
    return loudness

def peak_normalize(soundscape_audio, event_audio_list, peak_db=0.0):
    """
    Compute the scale factor required to peak normalize the audio such that
    max(abs(soundscape_audio)) = 1 (in case of peak_db=0).

    Parameters
    ----------
    soundscape_audio : np.ndarray
        The soundscape audio.
    event_audio_list : list
        List of np.ndarrays containing the audio samples of each isolated
        foreground event.
    peak_db: float
        The target peak level in dBFS. Default is 0 dBFS.

    Returns
    -------
    scaled_soundscape_audio : np.ndarray
        The peak normalized soundscape audio.
    scaled_event_audio_list : list
        List of np.ndarrays containing the scaled audio samples of
        each isolated foreground event. All events are scaled by scale_factor.
    scale_factor : float
        The scale factor used to peak normalize the soundscape audio.
    """
    eps = 1e-10
    target_peak = 10 ** (peak_db / 20.0)
    max_sample = np.max(np.abs(soundscape_audio))
    scale_factor = target_peak / (max_sample + eps)

    # scale the event audio and the soundscape audio:
    scaled_soundscape_audio = soundscape_audio * scale_factor

    scaled_event_audio_list = []
    for event_audio in event_audio_list:
        scaled_event_audio_list.append(event_audio * scale_factor)

    return scaled_soundscape_audio, scaled_event_audio_list, scale_factor

def peak_normalize_ambi(soundscape_audio, peak_db=0.0):
    """
    Compute the scale factor required to peak normalize the ambisonics signal such that the W channel is peaknormalized.
    i.e., max(abs(soundscape_audio[:, 0])) = 1.0.

    Parameters
    ----------
    soundscape_audio : np.ndarray
        The ambisonics soundscape audio.    
    peak_db: float
        The target peak level in dBFS. Default is 0 dBFS.

    Returns
    -------
    scaled_soundscape_audio : np.ndarray
        The peak normalized soundscape audio.    
    scale_factor : float
        The scale factor used to peak normalize the soundscape audio.
    """
    eps = 1e-10
    target_peak = 10 ** (peak_db / 20.0)
    max_sample = np.max(np.abs(soundscape_audio[:, 0]))
    scale_factor = target_peak  / (max_sample + eps)

    # scale the soundscape audio:
    scaled_soundscape_audio = soundscape_audio * scale_factor

    return scaled_soundscape_audio, scale_factor    

def  get_wavpack_duration(file_path):
    """ Get the duration of a .wv file """
    try:
        result = subprocess.run(['wvunpack', '-f', file_path], 
                               capture_output=True, text=True, check=True)
        parts = result.stdout.split(";")        
        fs = int(parts[0])        
        total_seconds = float(int(parts[5]) / fs)                
    except Exception as e:
        raise AmbiScaperError(f"Error reading WavPack duraction: {e}")                
    return total_seconds

def  get_wavpack_channelCount(file_path):
    """ Get the channel count of a .wv file """
    try:
        result = subprocess.run(['wvunpack', '-f4', file_path], 
                               capture_output=True, text=True, check=True)
    except Exception as e:
        raise AmbiScaperError(f"Error reading WavPack channel count: {e}")
    return int(result.stdout)        

def  get_wavpack_sampleRate(file_path):
    """ Get the sample rate of a .wv file """
    try:
        result = subprocess.run(['wvunpack', '-f1', file_path], 
                               capture_output=True, text=True, check=True)
    except Exception as e:
        raise AmbiScaperError(f"Error reading WavPack sample rate: {e}")
    return int(result.stdout)        

        