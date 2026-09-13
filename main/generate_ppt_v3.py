"""
Robust script to generate the presentation by clearing old text shapes
and adding explicitly sized, well-formatted new text boxes.
"""
from pptx import Presentation
from pptx.util import Inches, Pt
import os

template_path = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\PPT Format_SummerInternshp_Review2 (2).pptx'
output_path = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\main\RoadSense_Presentation_V4.pptx'

prs = Presentation(template_path)

def add_custom_text(slide, title_text, body_text):
    # Empty existing shapes to remove weird template placeholders
    for shape in slide.shapes:
        if shape.has_text_frame:
            shape.text_frame.clear()
            
    # Add Title Box
    title_box = slide.shapes.add_textbox(Inches(1), Inches(0.5), Inches(8), Inches(1))
    title_tf = title_box.text_frame
    title_tf.word_wrap = True
    p = title_tf.paragraphs[0]
    p.text = title_text
    p.font.name = 'Arial'
    p.font.size = Pt(36)
    p.font.bold = True
    
    # Add Body Box
    body_box = slide.shapes.add_textbox(Inches(1), Inches(1.8), Inches(8), Inches(5))
    body_tf = body_box.text_frame
    body_tf.word_wrap = True
    
    lines = body_text.split('\n')
    for i, line in enumerate(lines):
        if i == 0:
            p = body_tf.paragraphs[0]
        else:
            p = body_tf.add_paragraph()
            
        p.text = line
        p.font.name = 'Arial'
        p.font.size = Pt(24)

# Slide 0: Title Slide (requires a slightly different layout)
slide_0 = prs.slides[0]
for shape in slide_0.shapes:
    if shape.has_text_frame:
        shape.text_frame.clear()
        
title_0 = slide_0.shapes.add_textbox(Inches(1), Inches(2), Inches(8), Inches(1.5))
t0_tf = title_0.text_frame
t0_tf.word_wrap = True
p = t0_tf.paragraphs[0]
p.text = "ROADSENSE: AUTOMATED ROAD CONDITION SURVEYING VIA VISION AND SENSOR FUSION"
p.font.name = 'Arial'
p.font.size = Pt(36)
p.font.bold = True

body_0 = slide_0.shapes.add_textbox(Inches(1), Inches(3.5), Inches(8), Inches(3))
b0_tf = body_0.text_frame
b0_tf.word_wrap = True
b0_text = "Summer Internship (CSXXXX) Program B.Tech (H)\nReview 2\n\nTeam members: USN - Saket M\nProject Guide: Prof. Chidhananda R\nMentor: Prof Dattatreya P Mankame\nSchool of Computer Science and Engineering\nRV University, Bangalore"
for i, line in enumerate(b0_text.split('\n')):
    p = b0_tf.paragraphs[0] if i == 0 else b0_tf.add_paragraph()
    p.text = line
    p.font.name = 'Arial'
    p.font.size = Pt(20)

# Slide 2: Title and Problem definition
add_custom_text(prs.slides[2], 
                  "Problem Definition & Limitations", 
                  "Problem:\n• Manual surveys are slow, expensive, and highly subjective.\n• Potholes cause major accidents and damage, requiring rapid mapping.\n\nLimitations of Existing Systems:\n• Camera-only AI flags shadows and wet patches as false potholes.\n• Accelerometer-only systems confuse potholes with speed bumps.\n\nNovelty:\n• Simultaneous fusion of video and physics data.\n• A strict bidirectional verification gate with zero-cost edge computing.")

# Slide 3: Aim and Objectives
add_custom_text(prs.slides[3], 
                  "Aim and Objectives", 
                  "Aim:\nBuild a cheap, automated road monitoring system fusing visual and physical sensors.\n\nObjectives:\n• Collect simultaneous dashcam video and 10Hz telemetry.\n• Stabilise bumpy footage using optical flow and affine transformations.\n• Train a robust YOLOv8 model while solving catastrophic forgetting.\n• Compute vehicle-specific physical anomaly scores (Z-scores).\n• Implement a strict sensor-fusion logic gate.\n• Visualize verified data on a serverless web app.")

# Slide 4: Introduction & Scope
add_custom_text(prs.slides[4], 
                  "Introduction & Scope", 
                  "Introduction:\n• RoadSense combines YOLO object detection with smartphone accelerometers to create an automated, passive road surveyor.\n\nScope:\n• Focuses on urban asphalt roads (tested extensively in Bengaluru).\n• Covers the end-to-end pipeline from edge processing to web visualization.\n\nKey Challenges Addressed:\n• Severe shadow interference at 5 PM.\n• Catastrophic forgetting during AI retraining.\n• GPS drift causing map visualization failures.")

# Slide 5: Literature Survey
add_custom_text(prs.slides[5], 
                  "Literature Survey", 
                  "• Maeda et al. (2018): Camera-based AI. High precision but fails completely under shadows or wet roads.\n• Eriksson et al. (2008): Pothole Patrol (accelerometer). High recall but mistakes speed bumps for potholes.\n• Arya et al. (2021): Two-stage fusion in India. Sequential filtering instead of simultaneous cross-validation.\n\nResearch Gap:\n• Existing systems lack a strict, bidirectional verification gate where both modalities independently accept or reject each other. RoadSense fills this exact gap.")

# Slide 6: Requirement Specification
add_custom_text(prs.slides[6], 
                  "Requirement Specification", 
                  "Hardware Requirements:\n• Smartphone with 30fps camera, 3-axis accelerometer, and GPS.\n• Handlebar mount / Two-wheeler (Scooter).\n• Edge computing laptop for offline processing.\n\nSoftware Requirements:\n• Python 3.10+\n• Machine Learning: Ultralytics (YOLOv8), OpenCV, Pandas.\n• Web Dashboard: HTML/CSS/JS, Leaflet.js (OpenStreetMap).\n• Cloud: Kaggle (Free GPU training), Vercel (Free web hosting).")

# Slide 7: Proposed Methodology
add_custom_text(prs.slides[7], 
                  "Proposed Methodology", 
                  "Stage 1: Video Stabilisation\n• Optical flow feature tracking & Affine warps to smooth camera jitter.\n\nStage 2: Visual Detection\n• YOLOv8-nano runs on stabilized frames to detect dark patches (potholes).\n\nStage 3: Sensor Fusion (The Verification Gate)\n• Checking YOLO bounding boxes against accelerometer Z-scores.\n• Dropping visual detections if the road was physically smooth.\n\nStage 4: Web Visualization\n• Exporting the verified CSV to a serverless, dark-mode Leaflet map.")

# Slide 8: Implementation Work Completed
add_custom_text(prs.slides[8], 
                  "Implementation Work Completed", 
                  "• Data Collection App: Web app built to capture video & 10Hz telemetry simultaneously.\n• AI Training: Custom 'Mixed Dataset' created (combining Roboflow images + local shadow frames) and trained on Kaggle GPU.\n• Fusion Engine: process_ride.py script written for mathematical verification and Z-score calculation.\n• UI Dashboard: Vercel hosted map with a custom CSS-inverted Dark Mode for premium aesthetics.")

# Slide 9: Module-wise Implementation
add_custom_text(prs.slides[9], 
                  "Module-wise Implementation", 
                  "• Module 1: Vehicle Calibration (calibrate.py to establish baseline suspension physics per vehicle).\n• Module 2: Visual AI (Fine-tuned YOLOv8 neural network).\n• Module 3: Physics Engine (Z-score anomaly detection to find spikes > 1.5).\n• Module 4: Verification Gate (A strict AND gate to fuse visual & physical hits, massively reducing false positives).")

# Slide 10: Experimental Results / Output
add_custom_text(prs.slides[10], 
                  "Experimental Results & Output", 
                  "Testing on Ride 'test/four' (5,368 frames, 360 sensor rows):\n\n1. Baseline Model: 192 false detections (fooled by tree shadows).\n2. Shadow-only Retraining: 0 detections (Catastrophic forgetting - model went blind).\n3. Mixed Dataset + Fusion Gate: 35 perfectly verified potholes.\n\nResult:\nFalse positive reduction of >90% compared to camera-only detection. Every single logged pothole is backed by hard physical evidence.")

# Slide 11: Conclusion
add_custom_text(prs.slides[11], 
                  "Conclusion", 
                  "• Sensor fusion is absolutely mandatory; single-sensor systems are too fragile for real-world Indian roads.\n• Catastrophic forgetting is a major risk when fine-tuning AI on unbalanced datasets; a mixed dataset was the cure.\n• The strict verification gate successfully proved that off-the-shelf consumer smartphones can rival expensive dedicated survey vans when paired with smart edge-computing algorithms.")

# Slide 12: Future Work
add_custom_text(prs.slides[12], 
                  "Future Work", 
                  "• Mobile Edge Computing: Export YOLOv8 weights to CoreML/TFLite to run directly on the smartphone in real-time, eliminating the laptop processing step entirely.\n• Severity Grading: Rate potholes from 1 to 5 based on camera bounding box size and Z-score magnitude.\n• Crowdsourced Aggregation: Cluster reports from multiple riders via GPS. A pothole confirmed by 10 riders is highly trustworthy.\n• Municipal API Integration: Automatically pipe verified CSVs into BBMP's complaint system to generate work orders.")

# Slide 13: References
add_custom_text(prs.slides[13], 
                  "References", 
                  "[1] H. Maeda et al., 'Road damage detection and classification using deep neural networks with smartphone images,' Comput. Civ. Infrastruct. Eng., vol. 33, no. 12, pp. 1127-1141, 2018.\n\n[2] J. Eriksson et al., 'The Pothole Patrol: Using a mobile sensor network for road surface monitoring,' in Proc. 6th Int. Conf. Mobile Syst. Appl. Serv., 2008, pp. 29-39.\n\n[3] D. Arya et al., 'Deep learning-based road damage detection and classification for multiple countries,' Autom. Constr., vol. 132, 2021, Art. no. 103935.")

# Slide 14: Thank You
add_custom_text(prs.slides[14], 
                  "Questions?", 
                  "Thank you for your time.\n\nRoadSense: AI-Powered Road Infrastructure Intelligence")

prs.save(output_path)
print(f"Presentation saved to {output_path}")
