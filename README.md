# Video Summarization with YOLOv11 and Gemini

This project implements the video summarization system described in the research paper "Video Summarisation with Incident and Context Information using Generative AI" by De Silva et al.

## Features

- **Object Detection**: Uses YOLOv8 for real-time object detection
- **AI-Powered Analysis**: Leverages Google's Gemini AI for intelligent video description
- **Customizable Queries**: Support for specific incident detection and analysis
- **Frame-by-Frame Processing**: Intelligent frame sampling based on object detection
- **Comprehensive Reporting**: Detailed analysis with timestamps and confidence scores

## Installation

### Prerequisites

- Python 3.8 or higher
- Google GenAI API key (get from [Google AI Studio](https://aistudio.google.com/))
- Sufficient disk space for video processing

### Step 1: Clone or Download the Code

Save the main Python script as `video_summarizer.py`

### Step 2: Install Dependencies

```bash
uv pip install ultralytics google-genai pillow opencv-python tqdm numpy
```


### Step 3: Get Google GenAI API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Create a new API key
4. Copy the API key for use in the script

## Usage

### Basic Usage

```bash
python video_summarizer.py /path/to/video.mp4 --api-key YOUR_GEMINI_API_KEY
```

### Advanced Usage Examples

#### 1. Analyze for Specific Incidents

```bash
python video_summarizer.py video.mp4 --api-key YOUR_API_KEY --incident-type "accident"
```

#### 2. Custom Query Analysis

```bash
python video_summarizer.py video.mp4 --api-key YOUR_API_KEY --query "Identify any suspicious behavior or security concerns"
```

#### 3. Adjust Processing Parameters

```bash
python video_summarizer.py video.mp4 --api-key YOUR_API_KEY \
    --frame-rate 2.0 \
    --yolo-model yolov8s.pt \
    --target-classes person car motorcycle \
    --output results.json
```

#### 4. Fast Processing (Lower Frame Rate)

```bash
python video_summarizer.py video.mp4 --api-key YOUR_API_KEY --frame-rate 0.5
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--api-key` | Google GenAI API key (required) | - |
| `--output` | Output JSON file path | - |
| `--yolo-model` | YOLO model variant | `yolov11m.pt` |
| `--frame-rate` | Frames per second to process | `1.0` |
| `--target-classes` | Object classes to detect | `["person"]` |
| `--query` | Custom analysis prompt | - |
| `--incident-type` | Specific incident to analyze | - |
| `--temperature` | AI model creativity (0.0-1.0) | `0.1` |


## Supported Object Classes

The system can detect 80+ object classes including:
- person, car, motorcycle, bicycle, bus, truck
- traffic light, stop sign, bench, chair
- And many more (see COCO dataset classes)

## Example Output

```
============================================================
VIDEO ANALYSIS RESULTS
============================================================
Video: /path/to/traffic_video.mp4
Processing time: 45.23 seconds
Frames processed: 150
Relevant frames: 23
Incident frames found: 3

SUMMARY:
----------------------------------------
The video shows a busy road with various types of vehicles including cars, 
motorcycles, and buses. Heavy traffic is observed throughout most of the video. 
A motorcycle accident occurs around the 15-second mark, where a motorcyclist 
loses control and collides with a car. Emergency response and traffic 
management are visible in the later frames.

FRAME DETAILS:
----------------------------------------
Frame 450 (15.0s): A motorcycle accident is shown. A motorcyclist has fallen 
off his bike and is lying in the middle of the road...
Frame 540 (18.0s): Emergency responders are helping the injured motorcyclist...
```

## Performance Tips

1. **Frame Rate**: Lower frame rates (0.5-1.0 fps) are usually sufficient for most analyses
2. **Model Selection**: Use `yolov8n.pt` for speed, `yolov8s.pt` or `yolov8m.pt` for better accuracy
3. **Target Classes**: Specify only relevant classes to reduce false positives
4. **Video Quality**: Higher quality videos produce better AI descriptions

## Cost Considerations

- Google GenAI API charges per request
- Processing 1 hour of video at 1 fps ≈ 3600 API calls
- Estimate costs based on current Google AI pricing
- Use lower frame rates for cost optimization

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your Google GenAI API key is valid and has sufficient quota
2. **Video Loading Error**: Check video file format and path
3. **Memory Issues**: Reduce frame rate or use a smaller YOLO model
4. **Slow Processing**: Use faster YOLO model or increase frame interval

### Supported Video Formats

- MP4, AVI, MOV, MKV, and other common formats
- Recommended: MP4 with H.264 encoding

## Extending the System

The code is modular and can be extended for:

- Custom object detection models
- Different AI providers (OpenAI, Anthropic, etc.)
- Real-time video stream processing
- Integration with surveillance systems
- Custom incident detection algorithms

## Research Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{desilva2025video,
  title={Video Summarisation with Incident and Context Information using Generative AI},
  author={De Silva, Ulindu and Fernando, Leon and Bandara, Kalinga and Nawaratne, Rashmika},
  journal={arXiv preprint arXiv:2501.04764},
  year={2025}
}
```

## License

This implementation is provided for educational and research purposes. Please ensure compliance with:
- Google GenAI API terms of service
- Ultralytics YOLOv8 license
- Your local privacy and surveillance regulations

## Support

For technical issues:
1. Check the troubleshooting section
2. Verify all dependencies are installed correctly
3. Ensure your API key has sufficient quota
4. Review the original research paper for methodology details
