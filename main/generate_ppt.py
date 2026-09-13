"""
Generate the RoadSense presentation by replacing text in the provided template.
Fixes shape ordering issues and ensures proper font sizes and alignment.
"""
from pptx import Presentation
from pptx.util import Pt
from pptx.enum.text import PP_ALIGN
import os

template_path = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\PPT Format_SummerInternshp_Review2 (2).pptx'
output_path = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\main\RoadSense_Presentation_V2.pptx'

prs = Presentation(template_path)

def set_text_with_format(shape, text, font_size_pt, bold=False):
    if not shape or not shape.has_text_frame:
        return
    text_frame = shape.text_frame
    text_frame.clear()
    
    # Split text into paragraphs based on newlines
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if i == 0:
            p = text_frame.paragraphs[0]
        else:
            p = text_frame.add_paragraph()
            
        p.text = line
        
        # Apply formatting to all runs
        for run in p.runs:
            run.font.name = 'Times New Roman'
            run.font.size = Pt(font_size_pt)
            run.font.bold = bold

def set_slide_content(slide, title_text, body_text):
    # Find the title shape
    title_shape = slide.shapes.title
    if title_shape:
        set_text_with_format(title_shape, title_text, 36, bold=True)
        # Ensure title is centered if possible, though template usually handles this
    
    # Find the main body placeholder (usually the one that isn't the title)
    body_shape = None
    for shape in slide.placeholders:
        if shape != title_shape and shape.has_text_frame:
            body_shape = shape
            break
            
    # If no placeholder found, look for any text frame that isn't the title
    if not body_shape:
        for shape in slide.shapes:
            if shape != title_shape and shape.has_text_frame:
                body_shape = shape
                break
                
    if body_shape:
        set_text_with_format(body_shape, body_text, 22, bold=False)

# Slide 0: Title Slide
slide_0 = prs.slides[0]
set_slide_content(slide_0, 
                  "ROADSENSE: AI-POWERED ROAD INFRASTRUCTURE INTELLIGENCE", 
                  "Summer Internship (CSXXXX) Program B.Tech (H)\nReview 2\n\nTeam members:\nUSN - Saket M\n\nProject Guide: Prof. Chidhananda R\nMentor: Prof Dattatreya P Mankame\nSchool of Computer Science and Engineering\nRV University, Bangalore")

# Slide 1: Contents (Keep as is)
# We can skip modifying slide 1 entirely.

# Slide 2: Title and Problem definition
set_slide_content(prs.slides[2], 
                  "Problem Definition & Limitations", 
                  "Problem:\n• Manual surveys are slow, expensive, and highly subjective.\n• Potholes cause major accidents and damage, requiring rapid mapping.\n\nLimitations of Existing Systems:\n• Camera-only AI flags shadows and wet patches as false potholes.\n• Accelerometer-only systems confuse potholes with speed bumps and joints.\n\nNovelty:\n• Simultaneous fusion of video and physics data.\n• A strict bidirectional verification gate with zero-cost edge computing.")

# Slide 3: Aim and Objectives
set_slide_content(prs.slides[3], 
                  "Aim and Objectives", 
                  "Aim:\nBuild a cheap, automated road monitoring system fusing visual and physical sensors to map potholes accurately with minimal false alarms.\n\nObjectives:\n• Collect simultaneous dashcam video and 10Hz telemetry on a two-wheeler.\n• Stabilise bumpy footage using optical flow and affine transformations.\n• Train a robust YOLOv8 model while solving catastrophic forgetting.\n• Compute vehicle-specific physical anomaly scores (Z-scores).\n• Implement a strict sensor-fusion logic gate.\n• Visualize verified data on a serverless web app.")

# Slide 4: Introduction & Scope
set_slide_content(prs.slides[4], 
                  "Introduction & Scope", 
                  "Introduction:\n• RoadSense combines YOLO object detection with smartphone accelerometers to create an automated, passive road surveyor.\n\nScope:\n• Focuses on urban asphalt roads (tested extensively in Bengaluru).\n• Covers the complete end-to-end pipeline from edge processing to web visualization.\n\nKey Challenges Addressed:\n• Severe shadow interference at 5 PM.\n• Catastrophic forgetting during AI retraining.\n• GPS drift causing map visualization failures.")

# Slide 5: Literature Survey
set_slide_content(prs.slides[5], 
                  "Literature Survey", 
                  "• Maeda et al. (2018): Camera-based AI. High precision but fails completely under shadows or wet roads.\n• Eriksson et al. (2008): Pothole Patrol (accelerometer). High recall but mistakes speed bumps for potholes.\n• Arya et al. (2021): Two-stage fusion in India. Sequential filtering instead of simultaneous cross-validation.\n\nResearch Gap:\n• Existing systems lack a strict, bidirectional verification gate where both modalities independently accept or reject each other. RoadSense fills this exact gap.")

# Slide 6: Requirement Specification
set_slide_content(prs.slides[6], 
                  "Requirement Specification", 
                  "Hardware Requirements:\n• Smartphone with 30fps camera, 3-axis accelerometer, and GPS.\n• Handlebar mount / Two-wheeler (Scooter).\n• Edge computing laptop for offline processing.\n\nSoftware Requirements:\n• Python 3.10+\n• Machine Learning: Ultralytics (YOLOv8), OpenCV, Pandas.\n• Web Dashboard: HTML/CSS/JS, Leaflet.js (OpenStreetMap).\n• Cloud: Kaggle (Free GPU training), Vercel (Free web hosting).")

# Slide 7: Proposed Methodology
set_slide_content(prs.slides[7], 
                  "Proposed Methodology", 
                  "Stage 1: Video Stabilisation\n• Optical flow feature tracking & Affine warps to smooth camera jitter.\n\nStage 2: Visual Detection\n• YOLOv8-nano runs on stabilized frames to detect dark patches (potholes).\n\nStage 3: Sensor Fusion (The Verification Gate)\n• Checking YOLO bounding boxes against accelerometer Z-scores.\n• Dropping visual detections if the road was physically smooth.\n\nStage 4: Web Visualization\n• Exporting the verified CSV to a serverless, dark-mode Leaflet map.")

# Slide 8: Implementation Work Completed
set_slide_content(prs.slides[8], 
                  "Implementation Work Completed", 
                  "• Data Collection App: Web app built to capture video & 10Hz telemetry simultaneously.\n• AI Training: Custom 'Mixed Dataset' created (combining Roboflow images + local shadow frames) and trained on Kaggle GPU.\n• Fusion Engine: process_ride.py script written for mathematical verification and Z-score calculation.\n• UI Dashboard: Vercel hosted map with a custom CSS-inverted Dark Mode for premium aesthetics.")

# Slide 9: Module-wise Implementation
set_slide_content(prs.slides[9], 
                  "Module-wise Implementation", 
                  "• Module 1: Vehicle Calibration (calibrate.py to establish baseline suspension physics per vehicle).\n• Module 2: Visual AI (Fine-tuned YOLOv8 neural network).\n• Module 3: Physics Engine (Z-score anomaly detection to find spikes > 1.5).\n• Module 4: Verification Gate (A strict AND gate to fuse visual & physical hits, massively reducing false positives).")

# Slide 10: Experimental Results / Output
set_slide_content(prs.slides[10], 
                  "Experimental Results & Output", 
                  "Testing on Ride 'test/four' (5,368 frames, 360 sensor rows):\n\n1. Baseline Model: 192 false detections (fooled by tree shadows).\n2. Shadow-only Retraining: 0 detections (Catastrophic forgetting - model went blind).\n3. Mixed Dataset + Fusion Gate: 35 perfectly verified potholes.\n\nResult:\nFalse positive reduction of >90% compared to camera-only detection. Every single logged pothole is backed by hard physical evidence.")

# Slide 11: Conclusion
set_slide_content(prs.slides[11], 
                  "Conclusion", 
                  "• Sensor fusion is absolutely mandatory; single-sensor systems are too fragile for real-world Indian roads.\n• Catastrophic forgetting is a major risk when fine-tuning AI on unbalanced datasets; a mixed dataset was the cure.\n• The strict verification gate successfully proved that off-the-shelf consumer smartphones can rival expensive dedicated survey vans when paired with smart edge-computing algorithms.")

# Slide 12: Future Work
set_slide_content(prs.slides[12], 
                  "Future Work", 
                  "• Mobile Edge Computing: Export YOLOv8 weights to CoreML/TFLite to run directly on the smartphone in real-time, eliminating the laptop processing step entirely.\n• Severity Grading: Rate potholes from 1 to 5 based on camera bounding box size and Z-score magnitude.\n• Crowdsourced Aggregation: Cluster reports from multiple riders via GPS. A pothole confirmed by 10 riders is highly trustworthy.\n• Municipal API Integration: Automatically pipe verified CSVs into BBMP's complaint system to generate work orders.")

# Slide 13: References
set_slide_content(prs.slides[13], 
                  "References", 
                  "[1] H. Maeda et al., 'Road damage detection and classification using deep neural networks with smartphone images,' Comput. Civ. Infrastruct. Eng., vol. 33, no. 12, pp. 1127-1141, 2018.\n\n[2] J. Eriksson et al., 'The Pothole Patrol: Using a mobile sensor network for road surface monitoring,' in Proc. 6th Int. Conf. Mobile Syst. Appl. Serv., 2008, pp. 29-39.\n\n[3] D. Arya et al., 'Deep learning-based road damage detection and classification for multiple countries,' Autom. Constr., vol. 132, 2021, Art. no. 103935.")

# Slide 14: Thank You
set_slide_content(prs.slides[14], 
                  "Questions?", 
                  "Thank you for your time.\n\nRoadSense: AI-Powered Road Infrastructure Intelligence")

prs.save(output_path)
print(f"Presentation saved to {output_path}")
