"""
Parallelized Video Summarization with Incident and Context Information using Generative AI
Implementation based on the research paper by De Silva et al.

This parallelized version uses:
- YOLOv8 (Ultralytics) for object detection
- Google GenAI SDK (latest) for text generation and analysis
- OpenCV for video processing
- ThreadPoolExecutor and ProcessPoolExecutor for parallelization
- Various supporting libraries for analysis
"""

import cv2
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import argparse
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial
import threading
from queue import Queue, Empty
import asyncio
import aiohttp
from threading import Lock

# Core dependencies
try:
    from ultralytics import YOLO
    from google import genai
    from PIL import Image
    import base64
    import io
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install with: pip install ultralytics google-generativeai pillow opencv-python tqdm numpy")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FrameInfo:
    """Data class to store frame information"""
    frame_number: int
    timestamp: float
    description: str
    has_target_object: bool
    confidence: float = 0.0

class ParallelFrameProcessor:
    """
    Handles parallel processing of frames for object detection
    """
    
    def __init__(self, yolo_model: str, target_class_ids: List[int], max_workers: int = None):
        self.yolo_model = yolo_model
        self.target_class_ids = target_class_ids
        self.max_workers = max_workers or min(8, multiprocessing.cpu_count())
    
    def detect_objects_batch(self, frames_batch: List[Tuple[int, float, np.ndarray]]) -> List[Tuple[int, float, bool, float]]:
        """
        Process a batch of frames for object detection
        
        Args:
            frames_batch: List of (frame_number, timestamp, frame) tuples
            
        Returns:
            List of (frame_number, timestamp, has_target, confidence) tuples
        """
        # Load YOLO model in worker process
        yolo = YOLO(self.yolo_model)
        
        results = []
        for frame_num, timestamp, frame in frames_batch:
            has_target, confidence = self._detect_objects_single(yolo, frame)
            results.append((frame_num, timestamp, has_target, confidence))
        
        return results
    
    def _detect_objects_single(self, yolo, frame: np.ndarray) -> Tuple[bool, float]:
        """Detect objects in a single frame"""
        results = yolo(frame, verbose=False)
        
        has_target = False
        max_confidence = 0.0
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if class_id in self.target_class_ids:
                        has_target = True
                        max_confidence = max(max_confidence, confidence)
        
        return has_target, max_confidence
    
    def process_frames_parallel(self, frames: List[Tuple[int, float, np.ndarray]]) -> List[Tuple[int, float, bool, float]]:
        """
        Process all frames in parallel using multiprocessing
        
        Args:  
            frames: List of (frame_number, timestamp, frame) tuples
            
        Returns:
            List of (frame_number, timestamp, has_target, confidence) tuples
        """
        # Split frames into batches for processing
        batch_size = max(1, len(frames) // (self.max_workers * 2))
        frame_batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]
        
        logger.info(f"Processing {len(frames)} frames in {len(frame_batches)} batches using {self.max_workers} workers")
        
        all_results = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self.detect_objects_batch, batch): batch 
                for batch in frame_batches
            }
            
            # Collect results with progress bar
            with tqdm(total=len(frame_batches), desc="Object detection") as pbar:
                for future in as_completed(future_to_batch):
                    try:
                        batch_results = future.result()
                        all_results.extend(batch_results)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error processing batch: {e}")
                        pbar.update(1)
        
        # Sort results by frame number to maintain order
        all_results.sort(key=lambda x: x[0])
        return all_results

class GeminiAPIManager:
    """
    Manages parallel API calls to Gemini with rate limiting and retries
    """
    
    def __init__(self, api_key: str, max_concurrent: int = 5, rate_limit_per_minute: int = 60):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.max_concurrent = max_concurrent
        self.rate_limit_per_minute = rate_limit_per_minute
        self.request_times = Queue()
        self.lock = Lock()
        
    def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        with self.lock:
            current_time = time.time()
            
            # Remove requests older than 1 minute
            temp_times = []
            while not self.request_times.empty():
                try:
                    req_time = self.request_times.get_nowait()
                    if current_time - req_time < 60:  # Within last minute
                        temp_times.append(req_time)
                except Empty:
                    break
            
            # Put back recent requests
            for req_time in temp_times:
                self.request_times.put(req_time)
            
            # Check if we can make another request
            if self.request_times.qsize() >= self.rate_limit_per_minute:
                oldest_time = min(temp_times) if temp_times else current_time
                sleep_time = 60 - (current_time - oldest_time) + 1
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
            
            # Record this request
            self.request_times.put(current_time)
    
    def describe_frame_with_retry(self, frame: np.ndarray, method: str = "indirect", retries: int = 3) -> str:
        """
        Generate description for a frame with retry logic
        
        Args:
            frame: Input frame as numpy array
            method: "direct" or "indirect" prompting method
            retries: Number of retry attempts
            
        Returns:
            Text description of the frame
        """
        
        for attempt in range(retries):
            try:
                self._check_rate_limit()
                
                # Convert frame to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                if method == "direct":
                    prompt = """Describe if there is an accident or incident happening in this image. 
                    Be specific about what you observe and focus on any unusual activities or safety concerns."""
                else:  # indirect method
                    prompt = """Describe all the happenings in this image in detail. 
                    Include information about people, vehicles, activities, and any notable events or situations."""
                
                response = self.model.generate_content([prompt, pil_image])
                return response.text.strip()
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for frame description: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All attempts failed for frame description")
                    return "Unable to describe frame - API error"
    
    def process_frames_parallel(self, frames_data: List[Tuple[int, float, np.ndarray]]) -> List[FrameInfo]:
        """
        Process multiple frames in parallel using ThreadPoolExecutor
        
        Args:
            frames_data: List of (frame_number, timestamp, frame) tuples
            
        Returns:
            List of FrameInfo objects
        """
        
        frame_info_list = []
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit all description tasks
            future_to_frame = {}
            for frame_num, timestamp, frame in frames_data:
                future = executor.submit(self.describe_frame_with_retry, frame, "indirect")
                future_to_frame[future] = (frame_num, timestamp)
            
            # Collect results with progress bar
            with tqdm(total=len(frames_data), desc="Generating descriptions") as pbar:
                for future in as_completed(future_to_frame):
                    frame_num, timestamp = future_to_frame[future]
                    try:
                        description = future.result()
                        
                        frame_info = FrameInfo(
                            frame_number=frame_num,
                            timestamp=timestamp,
                            description=description,
                            has_target_object=True,
                            confidence=1.0  # Will be updated with actual confidence
                        )
                        frame_info_list.append(frame_info)
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        logger.error(f"Error processing frame {frame_num}: {e}")
                        pbar.update(1)
        
        # Sort by frame number to maintain order
        frame_info_list.sort(key=lambda x: x.frame_number)
        return frame_info_list

class ParallelVideoSummarizer:
    """
    Parallelized version of the Video Summarizer
    """
    
    def __init__(self, 
                 gemini_api_key: str,
                 yolo_model: str = "yolo11m.pt",
                 target_classes: List[str] = None,
                 frame_rate: float = 1.0,
                 temperature: float = 0.1,
                 max_detection_workers: int = None,
                 max_api_concurrent: int = 5,
                 api_rate_limit: int = 60):
        """
        Initialize the Parallel Video Summarizer
        
        Args:
            gemini_api_key: Google GenAI API key
            yolo_model: YOLOv8 model variant
            target_classes: List of object classes to detect
            frame_rate: Frames per second to process
            temperature: Temperature for Gemini model
            max_detection_workers: Max workers for object detection (default: CPU count)
            max_api_concurrent: Max concurrent API calls to Gemini
            api_rate_limit: API calls per minute limit
        """
        
        # Initialize Gemini API manager
        self.api_manager = GeminiAPIManager(
            api_key=gemini_api_key,
            max_concurrent=max_api_concurrent,
            rate_limit_per_minute=api_rate_limit
        )
        
        # Set up YOLO model and target classes
        logger.info(f"Loading YOLO model: {yolo_model}")
        yolo_temp = YOLO(yolo_model)  # Load once to get class names
        
        if target_classes is None:
            target_classes = ['person']
        self.target_classes = target_classes
        
        # Get class IDs from YOLO model
        self.target_class_ids = []
        class_names = yolo_temp.names
        for target_class in target_classes:
            for class_id, class_name in class_names.items():
                if target_class.lower() in class_name.lower():
                    self.target_class_ids.append(class_id)
                    logger.info(f"Tracking class: {class_name} (ID: {class_id})")
        
        if not self.target_class_ids:
            logger.warning(f"No matching classes found for {target_classes}. Using all detections.")
            self.target_class_ids = list(class_names.keys())
        
        # Initialize parallel frame processor
        self.frame_processor = ParallelFrameProcessor(
            yolo_model=yolo_model,
            target_class_ids=self.target_class_ids,
            max_workers=max_detection_workers
        )
        
        self.frame_rate = frame_rate
        self.temperature = temperature
        
    def extract_frames(self, video_path: str) -> List[Tuple[int, float, np.ndarray]]:
        """
        Extract frames from video based on specified frame rate
        
        Args:
            video_path: Path to input video
            
        Returns:
            List of tuples (frame_number, timestamp, frame_array)
        """
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"Video info: {fps:.2f} FPS, {total_frames} frames, {duration:.2f}s duration")
        
        # Calculate frame interval based on desired frame rate
        frame_interval = int(fps / self.frame_rate) if self.frame_rate < fps else 1
        
        frames = []
        frame_count = 0
        
        with tqdm(total=total_frames, desc="Extracting frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps
                    frames.append((frame_count, timestamp, frame.copy()))
                
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames for processing")
        return frames
    
    def analyze_video(self, video_path: str, query_prompt: str = None) -> Dict:
        """
        Main method to analyze video and generate summary using parallel processing
        
        Args:
            video_path: Path to input video
            query_prompt: Optional custom prompt for specific analysis
            
        Returns:
            Dictionary containing analysis results
        """
        
        logger.info(f"Starting parallel video analysis: {video_path}")
        start_time = time.time()
        
        # Step 1: Extract frames
        frames = self.extract_frames(video_path)
        
        # Step 2: Parallel object detection
        logger.info("Starting parallel object detection...")
        detection_results = self.frame_processor.process_frames_parallel(frames)
        
        # Step 3: Filter frames with target objects
        relevant_frames = []
        confidence_map = {}
        
        for i, (frame_num, timestamp, has_target, confidence) in enumerate(detection_results):
            if has_target:
                relevant_frames.append((frame_num, timestamp, frames[i][2]))  # frames[i][2] is the frame array
                confidence_map[frame_num] = confidence
        
        logger.info(f"Found {len(relevant_frames)} relevant frames out of {len(frames)} total")
        
        # Step 4: Parallel frame description generation
        if relevant_frames:
            logger.info("Starting parallel frame description generation...")
            frame_info_list = self.api_manager.process_frames_parallel(relevant_frames)
            
            # Update confidence scores
            for frame_info in frame_info_list:
                frame_info.confidence = confidence_map.get(frame_info.frame_number, 0.0)
        else:
            frame_info_list = []
        
        # Step 5: Generate final summary
        if frame_info_list:
            summary = self.generate_summary(frame_info_list, query_prompt)
        else:
            summary = "No relevant objects detected in the video."
        
        # Create detailed results
        processing_time = time.time() - start_time
        
        results = {
            'video_path': video_path,
            'processing_time': processing_time,
            'frames_processed': len(frames),
            'relevant_frames': len(frame_info_list),
            'frame_details': [
                {
                    'frame_number': fi.frame_number,
                    'timestamp': fi.timestamp,
                    'description': fi.description,
                    'confidence': fi.confidence
                } for fi in frame_info_list
            ],
            'summary': summary,
            'query_prompt': query_prompt,
            'parallelization_stats': {
                'detection_workers': self.frame_processor.max_workers,
                'api_concurrent': self.api_manager.max_concurrent,
                'total_frames': len(frames),
                'relevant_frames': len(frame_info_list)
            }
        }
        
        logger.info(f"Parallel analysis complete. Processing time: {processing_time:.2f}s")
        logger.info(f"Speedup estimation: ~{len(relevant_frames) / max(1, processing_time):.1f} frames/second")
        
        return results
    
    def generate_summary(self, frame_info_list: List[FrameInfo], query_prompt: str = None) -> str:
        """
        Generate final summary from frame descriptions
        
        Args:
            frame_info_list: List of FrameInfo objects
            query_prompt: Optional custom prompt for specific analysis
            
        Returns:
            Final summary text
        """
        
        # Combine all frame descriptions
        descriptions = []
        for fi in frame_info_list:
            descriptions.append(f"Frame {fi.frame_number} (t={fi.timestamp:.2f}s): {fi.description}")
        
        combined_text = "\n".join(descriptions)
        
        # Create summary prompt
        if query_prompt:
            prompt = f"""These are frame-wise descriptions of a video. 
            Analyze them according to this specific query: {query_prompt}
            
            Frame descriptions:
            {combined_text}
            
            Provide a comprehensive analysis focusing on the query."""
        else:
            prompt = f"""These are frame-wise descriptions of a video. 
            Understand, remove redundant information and provide a comprehensive summary 
            highlighting key incidents and events.
            
            Frame descriptions:
            {combined_text}
            
            Summary:"""
        
        try:
            response = self.api_manager.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Unable to generate summary due to API error."
    
    def analyze_for_specific_incidents(self, video_path: str, incident_type: str) -> Dict:
        """
        Analyze video for specific types of incidents using parallel processing
        
        Args:
            video_path: Path to input video
            incident_type: Type of incident to look for
            
        Returns:
            Dictionary containing incident analysis
        """
        
        query_prompt = f"Identify and describe any {incident_type} incidents in the video. Provide details about when they occur and what happens."
        
        results = self.analyze_video(video_path, query_prompt)
        
        # Additional incident-specific analysis using parallel processing
        incident_analysis_tasks = []
        
        with ThreadPoolExecutor(max_workers=self.api_manager.max_concurrent) as executor:
            futures = []
            for frame_detail in results['frame_details']:
                future = executor.submit(
                    self._analyze_incident_frame,
                    frame_detail['description'],
                    incident_type
                )
                futures.append((future, frame_detail))
            
            incident_frames = []
            for future, frame_detail in futures:
                try:
                    incident_response = future.result()
                    if "yes" in incident_response.lower():
                        incident_frames.append({
                            'frame_number': frame_detail['frame_number'],
                            'timestamp': frame_detail['timestamp'],
                            'description': frame_detail['description'],
                            'incident_analysis': incident_response
                        })
                except Exception as e:
                    logger.error(f"Error analyzing incident for frame {frame_detail['frame_number']}: {e}")
        
        results['incident_type'] = incident_type
        results['incident_frames'] = incident_frames
        results['incident_count'] = len(incident_frames)
        
        return results
    
    def _analyze_incident_frame(self, description: str, incident_type: str) -> str:
        """Helper method for incident analysis"""
        try:
            self.api_manager._check_rate_limit()
            response = self.api_manager.model.generate_content(
                f"Does this description indicate a {incident_type}? Answer yes/no and explain briefly: {description}"
            )
            return response.text
        except Exception as e:
            logger.error(f"Error in incident analysis: {e}")
            return "Analysis failed"

def main():
    """Main function with CLI interface for parallel processing"""
    
    parser = argparse.ArgumentParser(description="Parallel Video Summarization with YOLOv8 and Gemini")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--api-key", required=True, help="Google GenAI API key")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="YOLO model variant")
    parser.add_argument("--frame-rate", type=float, default=1.0, help="Frames per second to process")
    parser.add_argument("--target-classes", nargs="+", default=["person"], help="Object classes to detect")
    parser.add_argument("--query", help="Specific query prompt for analysis")
    parser.add_argument("--incident-type", help="Specific incident type to analyze")
    parser.add_argument("--temperature", type=float, default=0.1, help="Gemini model temperature")
    
    # Parallelization parameters
    parser.add_argument("--detection-workers", type=int, help="Max workers for object detection")
    parser.add_argument("--api-concurrent", type=int, default=5, help="Max concurrent API calls")
    parser.add_argument("--api-rate-limit", type=int, default=60, help="API calls per minute limit")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        return
    
    # Initialize parallel summarizer
    try:
        summarizer = ParallelVideoSummarizer(
            gemini_api_key=args.api_key,
            yolo_model=args.yolo_model,
            target_classes=args.target_classes,
            frame_rate=args.frame_rate,
            temperature=args.temperature,
            max_detection_workers=args.detection_workers,
            max_api_concurrent=args.api_concurrent,
            api_rate_limit=args.api_rate_limit
        )
    except Exception as e:
        logger.error(f"Failed to initialize parallel summarizer: {e}")
        return
    
    # Analyze video with parallel processing
    try:
        if args.incident_type:
            results = summarizer.analyze_for_specific_incidents(args.video_path, args.incident_type)
        else:
            results = summarizer.analyze_video(args.video_path, args.query)
        
        # Print results
        print(f"\n{'='*60}")
        print("PARALLEL VIDEO ANALYSIS RESULTS")
        print(f"{'='*60}")
        print(f"Video: {results['video_path']}")
        print(f"Processing time: {results['processing_time']:.2f} seconds")
        print(f"Frames processed: {results['frames_processed']}")
        print(f"Relevant frames: {results['relevant_frames']}")
        
        # Print parallelization stats
        if 'parallelization_stats' in results:
            stats = results['parallelization_stats']
            print(f"Detection workers: {stats['detection_workers']}")
            print(f"API concurrent calls: {stats['api_concurrent']}")
            print(f"Processing rate: ~{stats['relevant_frames'] / max(1, results['processing_time']):.1f} relevant frames/second")
        
        if args.incident_type and 'incident_count' in results:
            print(f"Incident frames found: {results['incident_count']}")
        
        print(f"\nSUMMARY:")
        print("-" * 40)
        print(results['summary'])
        
        if results['frame_details']:
            print(f"\nFRAME DETAILS:")
            print("-" * 40)
            for detail in results['frame_details']:
                print(f"Frame {detail['frame_number']} ({detail['timestamp']:.2f}s): {detail['description']}")
        
        # Save results if output path specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {args.output}")
            
    except Exception as e:
        logger.error(f"Parallel analysis failed: {e}")
        return

if __name__ == "__main__":
    main()