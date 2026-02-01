### Installation
```bash
# Create environment
python -m venv noodle_classifier_env
# OR conda
conda create -n noodle_classifier python=3.12

# Install dependencies
pip install -r requirements.txt
jupyter kernelspec install-self --user --name=noodle_classifier
```

### Contents: 
- AI Declaration Form v1.0.pdf
- cup_noodles_detector.ipynb
- PresentationSlides.pdf
- PresentationVideo.mp4
- requirements.txt
- video_to_jpg.py       # converts videos into jpg
- /models               # two ready-to-use models
- /images_to_test       # images that can be uploaded to test the model as part of method 2
- /data                 # please put the dataset in here