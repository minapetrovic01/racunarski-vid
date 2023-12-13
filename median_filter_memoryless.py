import numpy as np

def median_filter(signal, window_size):
    filtered_signal = np.zeros_like(signal)
    half_window = window_size // 2
    
    for i in range(len(signal)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(signal), i + half_window + 1)
        window = signal[start_idx:end_idx]
        filtered_signal[i] = np.median(window)
    
    return filtered_signal

# Example usage
input_signal = np.array([1, 3, 2, 5, 4, 6, 8, 7, 9])
window_size = 3

output_signal = median_filter(input_signal, window_size)
print("Input Signal:", input_signal)
print("Output Signal:", output_signal)
