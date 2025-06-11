import os
import cv2
from google import genai
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import base64
import time

sys_prompt = """"
    You are an intelligent scene analyzer. 
    Based on the following scene updates throughout the video, summarize the entire scene. 
    Identify if the scene looks like an accident, theft, sports, crowd, or general. 
    Avoid adding unnecessary words.
"""

GEMINI_API_KEY = "AIzaSyACa-SSJWU6zst5VfK_HZl1q29jY5VxNe4"

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it or uncomment and assign directly in the script.")

client = genai.Client(api_key = GEMINI_API_KEY)

gemini_pro_vision_model = genai.GenerativeModel('gemini-pro-vision')
gemini_pro_model = genai.GenerativeModel('gemini-2.0-flash')

yolo_model = YOLO('yolo11m.pt')

def image_to_base64(image_np):
    """Converts NumPY array image to base64 string."""
    pil_image = Image.fromarry(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def analyse_image_with_gemini_vision(image_np,prompt_text = sys_prompt):
    """
    Analyzes an image using Gemini Pro Vision model.
    image_np: NumPy array representing the image (OpenCV format).
    prompt_text: The prompt for Gemini Pro Vision.
    Returns the generated text description.
    """
    try:
        image_part = {
            "mime_type": "image/jpeg",
            "data" : BytesIO(cv2.imencode('.jpg', image_np)[1].tobytes())
        }
        
        generation_config = {
            "temperature": 0.11,
            "top_p" : 0.9,
            "top_k" : 40,
        }
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        ]
        
        contents = [image_part, prompt_text]
        
        response = gemini_pro_vision_model.generate_content(
            contents=contents,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream = False
        )
        
        return response.text.strip()
    
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return None
    
    
def summarize_text_with_gemini_pro(text_list, final_prompt):
    """
    Summarizes a list of text descriptions using Gemini Pro model.
    text_list: A list of strings, each being a frame description.
    final_prompt: The overarching prompt for the summary.
    Returns the summarized text.
    """
    
    combined_text = "\n".join(text_list)
    
    full_prompt = f"These are descriptions of events observed in a video. Consolidate, remove redundant information, and provide a concise summary focusing on key incidents. {final_prompt}\n\nDescriptions:\n{combined_text}"
    
    try:
        response = gemini_pro_model.generate_content(
            contents=[full_prompt],
            generation_config={
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 40
            },
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"}
            ],
            stream=False
        )
        
        return response.text.strip()
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return None
    
def process_video_for_summarization(
    video_path,
    output_summary_file="video_summary.txt",
    frame_rate_sampling=1, # Process 1 frame per second [1]
    target_objects=[], # List of objects to detect, e.g., ['person', 'car']
    vision_prompt="Describe everything happening in this image.",
    summary_prompt="What are the key incidents that occurred in the video?",
    temp_frame_dir="temp_frames"
):
    """
    Main function to process a video, identify incidents, and generate a summary.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # Create a temporary directory for frames if needed
    os.makedirs(temp_frame_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Could not get FPS from video. Exiting.")
        return

    frame_interval = int(fps / frame_rate_sampling) if frame_rate_sampling > 0 else 1
    if frame_interval < 1:
        frame_interval = 1 # Ensure at least one frame is processed if fps is very low

    frame_count = 0
    processed_frame_details = []
    
    print(f"Processing video: {video_path}")
    print(f"Video FPS: {fps}, Sampling 1 frame every {frame_interval} frames (approx {frame_rate_sampling} FPS)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        if frame_count % frame_interval == 0:
            current_time = frame_count / fps
            timestamp_str = time.strftime("%H:%M:%S", time.gmtime(current_time))
            print(f"Analyzing frame {frame_count} at {timestamp_str}...")

            # Step 1: Object Detection with YOLOv8 [1]
            yolo_results = yolo_model.predict(frame, verbose=False) # verbose=False to suppress output for each frame
            
            # Check if any of the target objects are detected
            detected_objects_in_frame = []
            if target_objects:
                for r in yolo_results:
                    for c in r.boxes.cls: # Iterate through detected classes
                        class_name = yolo_model.names[int(c)]
                        if class_name in target_objects:
                            detected_objects_in_frame.append(class_name)
            
            # If target_objects is empty, process all frames.
            # Otherwise, only process frames where specified objects are found.
            should_process_with_gemini = not target_objects or len(detected_objects_in_frame) > 0

            if should_process_with_gemini:
                # Step 2: Image Description with Gemini Pro Vision [1]
                frame_description = analyse_image_with_gemini_vision(frame, vision_prompt)
                if frame_description:
                    processed_frame_details.append(f"Timestamp {timestamp_str} (Frame {frame_count}): {frame_description}")
                    print(f"  - Gemini Vision Description: {frame_description[:100]}...") # Print first 100 chars
                else:
                    print(f"  - Failed to get Gemini Vision description for frame {frame_count}.")
            else:
                print(f"  - No target objects ({', '.join(target_objects)}) detected. Skipping Gemini Vision analysis.")

        frame_count += 1

    cap.release()
    print("\n--- Frame Analysis Complete ---")

    # Step 3: Text Summarization with Gemini Pro [1]
    if processed_frame_details:
        print("Generating final summary with Gemini Pro...")
        final_summary = summarize_text_with_gemini_pro(processed_frame_details, summary_prompt)
        
        if final_summary:
            with open(output_summary_file, "w") as f:
                f.write("--- Video Summary ---\n\n")
                f.write(f"Source Video: {video_path}\n")
                f.write(f"Frame Sample Rate (FPS): {frame_rate_sampling}\n")
                f.write(f"Target Objects for Detection: {', '.join(target_objects) if target_objects else 'All objects'}\n\n")
                f.write("--- Detailed Frame Descriptions (Filtered) ---\n")
                for detail in processed_frame_details:
                    f.write(detail + "\n")
                f.write("\n--- Final Summarized Incidents ---\n")
                f.write(final_summary)
            print(f"\nSummary saved to {output_summary_file}")
            print("\nGenerated Summary:")
            print(final_summary)
        else:
            print("Failed to generate final summary.")
    else:
        print("No relevant frames were processed to generate a summary.")

    # Clean up temporary frames (optional, if you saved them to disk)
    # import shutil
    # if os.path.exists(temp_frame_dir):
    #     shutil.rmtree(temp_frame_dir)
    #     print(f"Cleaned up temporary directory: {temp_frame_dir}")


if __name__ == "__main__":
    # 1. Video with a general summary
    # process_video_for_summarization(
    #     video_path="path/to/your/video.mp4",
    #     output_summary_file="traffic_summary.txt",
    #     frame_rate_sampling=0.5, # Process 1 frame every 2 seconds
    #     target_objects=[], # No specific object filter, describe everything
    #     vision_prompt="Describe the scene in detail, including all moving objects and general conditions.",
    #     summary_prompt="Provide a detailed overview of the video's content, highlighting major events."
    # )

    # 2. Video focusing on specific incidents (e.g., accidents involving cars or persons)
    process_video_for_summarization(
        video_path="HitNRun2-VEED.mp4", # <<< IMPORTANT: CHANGE THIS PATH
        output_summary_file="accident_incident_summary.txt",
        frame_rate_sampling=1, # Process 1 frame per second
        target_objects=['person', 'car', 'bus', 'motorcycle', 'truck'], # Objects to look for [12]
        vision_prompt="Analyze this image for any unusual activities, potential incidents, or interactions between vehicles and pedestrians. Be very specific about any accidents or unusual events.",
        summary_prompt="Identify and summarize all key incidents involving vehicles or people. State the timestamp of each incident and provide a concise description."
    )

    print("\n--- Project Execution Complete ---")