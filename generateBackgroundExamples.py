import argparse
import numpy as np
import soundfile as sf
import spaudiopy

def generate_colored_noise(num_samples, num_channels, noise_color='pink'):
    """
    Generates multichannel colored noise using FFT spectral scaling.
    Colors: 'white', 'pink', 'brown', 'blue'.
    """
    noise_color = noise_color.lower()
    
    # 1. Generate White Noise (Base)
    # Shape: (Num_Channels, Num_Samples)
    white = np.random.randn(num_channels, num_samples)
    
    # Optimization: If white is requested, return immediately
    if noise_color == 'white':
        return white
    
    # 2. Transform to Frequency Domain
    X = np.fft.rfft(white, axis=1)
    
    # 3. Create Frequency Vector
    freqs = np.fft.rfftfreq(num_samples)
    
    # 4. Determine Scaling Factor based on Color
    # We use np.errstate to safely handle division by zero at DC (index 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        if noise_color == 'pink':
            # Power ~ 1/f -> Amplitude ~ 1/sqrt(f)
            scale = 1.0 / np.sqrt(freqs)
        elif noise_color == 'brown':
            # Power ~ 1/f^2 -> Amplitude ~ 1/f
            scale = 1.0 / freqs
        elif noise_color == 'blue':
            # Power ~ f -> Amplitude ~ sqrt(f)
            scale = np.sqrt(freqs)
        else:
            raise ValueError(f"Unknown noise type: {noise_color}")

    # 5. Handle DC and Nyquist edge cases
    scale[0] = 0.0  # Remove DC component to prevent drift
    if np.isinf(scale).any():
        scale[np.isinf(scale)] = 0.0

    # 6. Apply Spectral Scaling and IFFT
    return np.fft.irfft(X * scale, axis=1)

def main(order, duration, fs, filename, noise_type):
    print(f"--- Generating {order}th Order Diffuse {noise_type.title()} Noise ---")
    #print(f"Duration: {duration}s | SR: {fs}Hz")
    
    num_samples = int(fs * duration)
    
    # --- 1. Setup Virtual Source Grid (The Diffuse Field) ---
    # T-design degree should be >= 2*Order + 1
    degree = 21    
    vecs = spaudiopy.grids.load_t_design(degree=degree)
    
    num_virtual_sources = vecs.shape[0]
    
    # Convert to spherical coordinates
    azi, zen, r = spaudiopy.utils.cart2sph(vecs[:, 0], vecs[:, 1], vecs[:, 2])

    # --- 2. Generate Multichannel Noise ---    
    src_signals = generate_colored_noise(num_samples, num_virtual_sources, noise_type)

    # --- 3. Encode to Ambisonics (AmbiX) ---    
    # Generate SH Matrix
    Y_nm = spaudiopy.sph.sh_matrix(order, azi, zen, sh_type='real')
    
    # Matrix Multiplication: (Channels, Sources) @ (Sources, Samples)
    ambisonics_sig = Y_nm.T @ src_signals

    # --- 4. Normalize and Save ---    
           
    peak = np.max(np.abs(ambisonics_sig))
    # Normalize to -1.0 dBFS
    target_dbfs = -1.0 
    target_linear = 10 ** (target_dbfs / 20.0)
    if peak > target_linear:        
        ambisonics_sig *= (target_linear / peak)
        
    sf.write(filename, ambisonics_sig.T, fs)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Diffuse Ambisonics Noise Field")
    
    # Arguments
    parser.add_argument('-t', '--type', type=str, default='pink', 
                        choices=['white', 'pink', 'brown', 'blue'],
                        help='Noise color type (default: pink)')
    parser.add_argument('-o', '--order', type=int, default=5, 
                        help='Ambisonics Order (default: 5)')
    parser.add_argument('-d', '--duration', type=float, default=15.0, 
                        help='Duration in seconds (default: 15.0)')
    parser.add_argument('-sr', '--samplerate', type=int, default=48000, 
                        help='Sampling Rate in Hz (default: 48000)')
    parser.add_argument('-f', '--filename', type=str, default=None, 
                        help='Output filename. If ignored, auto-names based on params.')

    args = parser.parse_args()
    
    # Auto-generate filename if not provided
    if args.filename is None:
        args.filename = f"diffuse_{args.type}_{args.order}o_{int(args.duration)}s.wav"

    main(args.order, args.duration, args.samplerate, args.filename, args.type)