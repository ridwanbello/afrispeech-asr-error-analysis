from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import re


normalizer = BasicTextNormalizer()

SPOKEN_PUNCT = [
    r'\bfull stop\b', r'\bcomma\b', r'\bperiod\b',
    r'\bcolon\b', r'\bsemicolon\b', r'\bsemi colon\b',
    r'\bquestion mark\b', r'\bexclamation mark\b',
    r'\bhyphen\b', r'\bnew line\b', r'\bnext line\b',
    r'\bopen bracket\b', r'\bclose bracket\b',
    r'\bopen parenthesis\b', r'\bclose parenthesis\b',
    r'\bslash\b', r'\bplus sign\b',
]

def normalize(text):
    text = str(text).lower()
    for pattern in SPOKEN_PUNCT:
        text = re.sub(pattern, '', text)
    text = normalizer(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def estimate_snr(audio):
    """
    Energy-based SNR estimation using VAD.
    Splits signal into speech and non-speech frames,
    estimates SNR from energy ratio.
    """
    import numpy as np
    
    audio = np.array(audio, dtype=np.float64)
    
    if len(audio) == 0 or np.max(np.abs(audio)) == 0:
        return None
    
    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-10)
    
    # Frame the signal into 20ms frames at 16kHz = 320 samples
    frame_size = 320
    frames = [audio[i:i+frame_size] for i in range(0, len(audio)-frame_size, frame_size)]
    
    if len(frames) < 2:
        return None
    
    # Energy per frame
    energies = np.array([np.sum(f**2) / len(f) for f in frames])
    
    # Simple VAD: top 70% energy frames = speech, bottom 30% = noise
    threshold = np.percentile(energies, 30)
    speech_energy = np.mean(energies[energies >= threshold])
    noise_energy  = np.mean(energies[energies < threshold])
    
    if noise_energy == 0:
        return 40.0  # very clean
    
    snr = 10 * np.log10(speech_energy / noise_energy)
    return float(snr)