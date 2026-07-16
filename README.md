# Handgesture Detection for People in Danger

A real-time hand gesture recognition system built for a Computer Vision course project. The system detects a clenched-fist gesture as a distress/help signal and triggers a warning alert, aimed at helping people silently signal that they are in danger.

## Problem Statement

People in threatening or unsafe situations often can't call for help out loud — speaking or reaching for a phone can escalate danger. This project explores whether a simple, recognizable hand gesture (a clenched fist) can be detected reliably enough via webcam to trigger an automated warning, giving people a discreet way to signal distress.

## How It Works

1. **Hand detection** — the system captures webcam frames and detects the hand region and key landmarks.
2. **Gesture classification** — the detected hand pose is classified to determine whether it matches the "clenched fist" distress gesture.
3. **Warning trigger** — when the fist gesture is detected with sufficient confidence, the system raises a warning signal.

> *(Fill in: which library did you use for hand landmark detection — MediaPipe? OpenCV alone? And what model/architecture did you use for classification — a CNN you trained, or a pretrained model you fine-tuned?)*

## Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- *(add MediaPipe or other libraries you used)*

## Dataset

*(Fill in: did you build your own dataset of fist vs. non-fist gestures? How many samples, how was it collected/labeled, and what preprocessing did you apply?)*

## Challenges & Debugging

This section matters most for demonstrating engineering judgment — a few examples to fill in based on what you actually ran into:

- **False positives/negatives** — did certain everyday gestures (e.g. a closed hand while holding something) get misclassified as the danger signal? How did you tune the confidence threshold or retrain to reduce this?
- **Lighting/background sensitivity** — did detection accuracy drop in poor lighting or cluttered backgrounds? What did you do about it?
- **Real-time performance** — did you hit any latency or frame-rate issues running detection live, and how did you optimize?

## Results

*(Fill in: accuracy/precision/recall if you measured them, or qualitative results — e.g. "correctly detected the fist gesture in X out of Y test trials")*

## How to Run

```bash
# Clone the repository
git clone https://github.com/adityanhh/handgestureapp-tensorflow
cd handgestureapp-tensorflow

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

*(Adjust the exact commands/filenames to match your actual repo structure)*

## Future Improvements

- Support for additional distress gestures beyond the clenched fist
- Sending real-time alerts to a designated contact or authority (SMS/notification integration)
- Improving robustness across lighting conditions and skin tones

## Author

Aditya Nugraha — Final-year student, National Institute of Technology (Itenas)
