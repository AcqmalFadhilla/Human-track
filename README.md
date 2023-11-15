<h1 align="center">
  Human Tracking
</h1>

## <div align="center">Quick Start Example</div>

```bash
git clone git@github.com:ozora-ogino/tflite-human-tracking.git
cd tflite-human-tracking
python main.py --src ./data/<YOUR_VIDEO_FILE>.mp4 --model ./models/<YOLOV5_MODEL>.tflite

# Set directions.
# For the value of direction you can choose one of 'bottom', 'top', right', 'left' or None.
python src/main.py --src ./data/trim10s.mp4 \
                   --model ./models/yolov5s-fp16.tflite \
                   --directions="{'total': None, 'inside': 'bottom', 'outside': 'top'}"
```

### Docker

```bash
./build_image.sh
./run.sh ./data/<YOUR_VIDEO_FILE>.mp4 ./models/<YOLOV5_MODEL>.tflite
```

di `outputs` folder.


### Dataset
[TownCentreXVID](https://www.kaggle.com/ashayajbani/oxford-town-centre/version/4?select=TownCentreXVID.mp4).
