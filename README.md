# Speech parameters analytics

### **Main Problem to solve:** most metrics are either used separately or used without any description and meaning of the score.

*For instance, if you get two metrics MOS 3.4 and SNR 3.4, there is no framework to get these metrics together and understand the values.* 

### **Main goal: to create a pipeline to get the values of the speech quality with both metrics description and the meaning of the scores.**

### Types of the Metrics

**Subjective human-perception metrics:** 

- **Perceptual Evaluation of Speech Quality (PESQ)**
    
    PESQ accepts only narrow-band input and is *not directly applicable* on other bandwidths.
    
    - Paired metric.
    
    ```json
        "PESQ": {
            "description": "Perceptual Evaluation of Speech Quality is a metric that employs a perceptual model that mimics human auditory perception. It takes into account factors such as time and frequency masking, where the human ear is less sensitive to certain distortions in the presence of louder sounds.",
            "low": "Below 2 indicates poor quality.",
            "high": "Above 2 indicates good quality with the values close to 5 as perfect sound"
        },
    ```

    
- **Mean Opinion Score (MOS):**
    - UnPaired metric.
    
    Subjective measure of overall speech quality rated by human listeners. 
    
    ```json
        "MOS": {
            "description": "Mean Opinion Score is a subjective metric that quantifies the overall perceived quality of audio or video signals. It is obtained through rigorous testing involving human listeners who rate the quality on a predefined scale, usually from 1 to 5.",
            "low": "Below 3 indicates poor quality.",
            "high": "Above 3 indicates good quality, with values close to 5 indicating excellent quality."
        },
    ```
    

**Objective metrics:** 

- **Short-Time Objective Intelligibility (STOI): Assesses speech intelligibility.**
    
    Intelligibility measure which is highly correlated with the intelligibility of degraded speech signals, e.g., due to additive noise, single/multi-channel noise reduction, binary masking and vocoded speech as in CI simulations.  
    
    ```json
        "STOI": {
            "description": "STOI evaluates the similarity between the clean (reference) speech signal and the processed (possibly noisy or enhanced) speech signal. The metric works on short-time segments of the speech signal. It computes the correlation between the clean and processed signals over overlapping short-time windows, typically 384 milliseconds in length.",
            "low": "0 indicates completely unintelligible speech.",
            "high": "1 indicates perfectly intelligible speech."
        },
    ```
        
    - Paired metric.
- **ESTOI (Extended Short-Time Objective Intelligibility):**
An extension of STOI that aims to provide better performance, especially for fluctuating noise conditions.
    - Paired metric.
    
    ```json
        "ESTOI": {
            "description": "eSTOI is designed to handle a wider range of distortion types, including those that are non-linear. eSTOI operates on short-time segments of the speech signal and uses a time-frequency representation. It compares the clean and processed speech signals within these short-time frames. eSTOI typically divides the speech signal into more frequency bands compared to STOI, allowing for a finer granularity in measuring the intelligibility across different parts of the spectrum.",
            "low": "0 indicates completely unintelligible speech.",
            "high": "1 indicates perfectly intelligible speech."
        },
    ```
    
- **Signal-to-Noise Ratio (SNR):** 
Signal-to-noise ratio (SNR) is a measure used in science and engineering that compares the level of a desired signal to the level of background noise.
    - UnPaired metric.
    
    ```json
        "SNR": {
            "description": "SNR is used to measure the clarity of a signal by comparing the level of the desired signal to the level of background noise. SNR is calculated as the ratio of the power of the signal to the power of the noise",
            "low": "Low value more noise relative to the signal, resulting in poorer quality. An SNR of 0 dB means the signal and noise power are equal, indicating poor quality.",
            "high": "High value indicates a clearer signal with less noise, thus better quality. For example, an SNR of 30 dB means the signal power is 1000 times the noise power, which is a high-quality signal."
        },
    ```
    
- **C50**
    - UnPaired metric.
    
    ```json
        "C50": {
            "description" : "C50 is defined as the ratio of early to late arriving sound energy, where early refers to the sound energy arriving within the first 50 milliseconds after the direct sound, and late refers to the sound energy arriving after this period.",
            "low" : "Low value signifies more reverberation and less clarity, as more sound energy arrives later, contributing to echoes and blurred speech perception.",
            "high": "High value indicates clearer speech perception, as a larger portion of the sound energy arrives early, contributing to direct and comprehensible speech sounds."
        },
    ```
    
- **Scale-Invariant-Signal-Distortion-Ratio**
    - Paired metric.

```json
    "Scale-Invariant-Signal-Distortion-Ratio": {
        "description": "Scale-Invariant Signal Distortion Ratio (SI-SDR) is a metric used to evaluate the performance of source separation algorithms. It measures the distortion of the separated signals in a scale-invariant manner, making it robust against variations in signal amplitude.",
        "low": "Low values indicate high distortion and poor separation quality.",
        "high": "High values indicate low distortion and better separation quality."
    }
```

Paired metrics: [’**Scale-Invariant-Signal-Distortion-Ratio’, ‘ESTOI’, ‘STOI’, ‘PESQ’**]

Unpaired metrics: [’**C50’, ‘SNR’, ‘MOS’**]
