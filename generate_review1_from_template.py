from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor

template_path = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\PPT Format_SummerInternshp_Review2 (2).pptx'
output_path = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\main\1RUA24CSE0395_ppt01_v2.pptx'

IMG_DIR = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\main\ppt_images'

import os

prs = Presentation(template_path)
slides = list(prs.slides)

def clear_and_set(shape, lines):
    """Clear all runs in a shape's text frame and set new lines, preserving first run's formatting."""
    if not shape.has_text_frame:
        return
    tf = shape.text_frame
    # Grab formatting from the very first run
    ref_font_name = 'Times New Roman'
    ref_font_size = Pt(14)
    ref_font_bold = False
    ref_font_color = None
    if tf.paragraphs and tf.paragraphs[0].runs:
        r0 = tf.paragraphs[0].runs[0]
        ref_font_name = r0.font.name or 'Times New Roman'
        ref_font_size = r0.font.size or Pt(14)
        ref_font_bold = r0.font.bold
        try:
            ref_font_color = r0.font.color.rgb
        except:
            ref_font_color = None

    # Clear everything
    for para in tf.paragraphs:
        for run in para.runs:
            run.text = ''

    # Write new lines
    for i, line in enumerate(lines):
        if i < len(tf.paragraphs):
            p = tf.paragraphs[i]
            if p.runs:
                p.runs[0].text = line
            else:
                run = p.add_run()
                run.text = line
                run.font.name = ref_font_name
                run.font.size = ref_font_size
                run.font.bold = ref_font_bold
                if ref_font_color:
                    run.font.color.rgb = ref_font_color
        else:
            p = tf.add_paragraph()
            run = p.add_run()
            run.text = line
            run.font.name = ref_font_name
            run.font.size = ref_font_size
            run.font.bold = ref_font_bold
            if ref_font_color:
                run.font.color.rgb = ref_font_color

def find_shape(slide, partial_name):
    for shape in slide.shapes:
        if partial_name in shape.name:
            return shape
    return None

# ========== SLIDE 1: Title ==========
s = slides[0]
shape = find_shape(s, 'Shape;60')
if shape:
    clear_and_set(shape, [
        '   RoadSense: Road Infrastructure',
        '  Monitoring Using Vibration Analysis',
        'B.Tech (Honours) - Computer Science',
        'Review 1',
        ''
    ])

shape = find_shape(s, 'Shape;64')
if shape:
    clear_and_set(shape, [
        '                                                      Team Members',
        '1RUA24CSE0395  - Saket M',
        '',
        '',
        '',
        ''
    ])

shape = find_shape(s, 'Shape;65')
if shape:
    clear_and_set(shape, [
        'Project Guide / Mentor: ',
        'Prof. Sayan Unankard',
        'Assistant Professor',
        'School of Computer Science and Engineering',
        'RV University, Bangalore'
    ])

shape = find_shape(s, 'Shape;67')
if shape:
    clear_and_set(shape, ['Internal'])

# ========== SLIDE 2: Contents ==========
s = slides[1]
shape = find_shape(s, 'Shape;73')
if shape:
    clear_and_set(shape, [
        'Title and Problem Definition',
        'Aim and Objectives',
        'Introduction & Scope',
        'Literature Survey',
        'Requirement Specification',
        'Proposed Methodology',
        'Implementation (Weeks 1–4)',
        'Results & Screenshots',
        'Future Work',
        'Conclusion',
        'References'
    ])

# ========== SLIDE 3: Title and Problem Definition ==========
s = slides[2]
shape = find_shape(s, 'Shape;79')
if shape:
    clear_and_set(shape, ['Title and Problem Definition', ''])

shape = find_shape(s, 'Shape;78')
if shape:
    clear_and_set(shape, [
        'Title: RoadSense — Road Infrastructure Monitoring Using Vibration Analysis',
        'India has 6M+ km of roads; potholes cause thousands of accidents annually.',
        'Manual road surveys are expensive, slow, and subjective — engineers walk the roads by hand.',
        'No affordable, automated system exists that uses smartphone vibrations to detect and map potholes.',
        'Goal: Use the accelerometer and gyroscope already inside every smartphone to detect road defects automatically.'
    ])

# ========== SLIDE 4: Aim and Objectives ==========
s = slides[3]
shape = find_shape(s, 'Shape;86')
if shape:
    clear_and_set(shape, ['Aim and Objectives', '', ''])

shape = find_shape(s, 'Shape;85')
if shape:
    clear_and_set(shape, [
        'Aim: Design an affordable, automated road-condition monitoring system using smartphone inertial sensors.',
        'Obj 1: Build a Progressive Web App (PWA) to record accelerometer, gyroscope, and GPS data simultaneously while riding.',
        'Obj 2: Develop a custom web-based data labeling tool for annotating time-series sensor CSVs.',
        'Obj 3: Collect real-world ride data on Bangalore roads and create a labeled ground-truth dataset.',
        'Obj 4: Establish a spike-detection algorithm using Z-axis acceleration thresholds.',
        'Obj 5: Visualize detected road anomalies on an interactive geographic map.'
    ])

# ========== SLIDE 5: Introduction & Scope ==========
s = slides[4]
shape = find_shape(s, 'Shape;93')
if shape:
    clear_and_set(shape, ['Introduction & Scope', ''])

shape = find_shape(s, 'Shape;92')
if shape:
    clear_and_set(shape, [
        'Road surface degradation is a critical public safety hazard in India.',
        'Smartphones contain powerful inertial measurement units (accelerometers + gyroscopes) that can sense road vibrations.',
        'When a vehicle hits a pothole, the Z-axis acceleration spikes sharply — this is the core signal we exploit.',
        'Scope: Mount a smartphone on a two-wheeler, record sensor data during rides, and automatically flag anomalies.',
        'Weeks 1–4 focused on building the data collection app and the labeling tool to create training datasets.'
    ])

# ========== SLIDE 6: Literature Survey ==========
s = slides[5]
shape = find_shape(s, 'Shape;99')
if shape:
    clear_and_set(shape, ['Literature Survey'])

shape = find_shape(s, 'Shape;100')
if shape:
    clear_and_set(shape, [
        'Eriksson et al. (2008) — "Pothole Patrol": Used taxi-mounted smartphone accelerometers. Achieved 80–90% recall but high false-positive rate from speed bumps.',
        'Mednis et al. (2011) — Defined Z-axis threshold-based anomaly detection on Android phones. Limitation: Fixed thresholds fail at varying vehicle speeds.',
        'Perttunen et al. (2011) — Used machine learning on accelerometer features for road classification. Limitation: Required extensive manual feature engineering.',
        'Tai et al. (2010) — Proposed adaptive threshold using vehicle speed profiles. Limitation: Tested only on cars, not two-wheelers.',
        'Research Gap: No existing system provides an end-to-end pipeline from data collection → labeling → detection → map visualization using just a smartphone.'
    ])

# ========== SLIDE 7: Requirement Specification ==========
s = slides[6]
shape = find_shape(s, 'Shape;107')
if shape:
    clear_and_set(shape, ['Requirement Specification'])

shape = find_shape(s, 'Shape;106')
if shape:
    clear_and_set(shape, [
        'Hardware: Any Android smartphone with accelerometer, gyroscope, and GPS. A two-wheeler vehicle for data collection.',
        'Software: HTML5, CSS3, JavaScript (Vanilla), Python 3.13.',
        'Libraries: Chart.js (interactive graphing), Papa Parse (CSV parsing), Leaflet.js (map rendering).',
        'Data Format: CSV files with columns for timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, latitude, longitude, speed.',
        'Deployment: GitHub Pages (zero-cost static hosting for the PWA and map).'
    ])

# ========== SLIDE 8: Proposed Methodology ==========
s = slides[7]
shape = find_shape(s, 'Shape;114')
if shape:
    clear_and_set(shape, ['Proposed Methodology'])

shape = find_shape(s, 'Shape;113')
if shape:
    clear_and_set(shape, [
        'Stage 1 — Data Collection: PWA records accelerometer, gyroscope, GPS, and speed at ~100 Hz while riding.',
        'Stage 2 — Data Labeling: Upload the CSV to the custom labeling tool. Visually identify Z-axis spikes and annotate them as "Pothole" or "Speed Breaker".',
        'Stage 3 — Anomaly Detection: Apply threshold-based spike detection on the Z-axis acceleration data. Flag readings where |accel_z| exceeds a configurable sensitivity threshold.',
        'Stage 4 — Map Visualization: Plot all flagged GPS coordinates on an interactive Leaflet.js map, color-coded by severity (green = smooth, yellow = degraded, red = terrible).',
        'Stage 5 — Export: Generate labeled CSV files with the annotation column for downstream ML training.'
    ])

# ========== SLIDE 9: Implementation — Timeline ==========
s = slides[8]
shape = find_shape(s, 'Shape;121')
if shape:
    clear_and_set(shape, ['Implementation Timeline (Weeks 1–4)'])

shape = find_shape(s, 'Shape;120')
if shape:
    clear_and_set(shape, [
        'Week 1: Research and planning. Studied existing pothole detection literature. Set up the development environment.',
        'Week 2: Built the RoadSense PWA — records accelerometer, gyroscope, GPS, and speed data simultaneously with a "Start Ride" button.',
        'Week 3: Developed the custom Labeling Tool web app — drag-and-drop CSV upload, interactive Chart.js graph, region selection, and CSV export with labels.',
        'Week 4: Conducted initial test rides on Bangalore roads. Collected and labeled 5 ride datasets (~22,000 data points). Identified key challenges (speed bumps vs potholes).'
    ])

# ========== SLIDE 10: Fig 1 — RoadSense App UI ==========
s = slides[9]
shape = find_shape(s, 'Shape;128')
if shape:
    clear_and_set(shape, ['Fig 1: RoadSense Data Collection App'])

shape = find_shape(s, 'Shape;127')
if shape:
    clear_and_set(shape, [
        'The RoadSense PWA running on a smartphone.',
        'Features: Configurable spike detection sensitivity (threshold slider), voice labeling, popup alerts for real-time event marking.',
        'The app records accelerometer, gyroscope, GPS coordinates, and vehicle speed simultaneously at ~100 Hz.',
        'Data is exported as a raw CSV file for offline analysis.'
    ])

# Add the app UI image
img_path = os.path.join(IMG_DIR, 'app_ui.png')
if os.path.exists(img_path):
    s.shapes.add_picture(img_path, Inches(0.5), Inches(3.8), height=Inches(3.2))

# ========== SLIDE 11: Fig 2 — Labeling Tool (Empty) ==========
s = slides[10]
shape = find_shape(s, 'Shape;135')
if shape:
    clear_and_set(shape, ['Fig 2: Labeling Tool — CSV Upload Interface'])

shape = find_shape(s, 'Shape;134')
if shape:
    clear_and_set(shape, [
        'The custom-built Labeling Tool web application.',
        'Users drag and drop a sensor CSV file or load demo data to visualize the acceleration graph.',
        'Right panel: Label Editor with Pothole (P) and Speed Breaker (S) buttons, selection width slider, and labeled regions list.',
        ''
    ])

img_path = os.path.join(IMG_DIR, 'labeling_empty.png')
if os.path.exists(img_path):
    s.shapes.add_picture(img_path, Inches(0.3), Inches(3.5), width=Inches(9.2))

# ========== SLIDE 12: Fig 3 — Labeling Tool (With Data) ==========
s = slides[11]
shape = find_shape(s, 'Shape;142')
if shape:
    clear_and_set(shape, ['Fig 3: Labeling Tool — Annotated Ride Data'])

shape = find_shape(s, 'Shape;141')
if shape:
    clear_and_set(shape, [
        'A real ride dataset (22,542 samples) loaded into the labeling tool.',
        'Blue line: Z-axis acceleration (vertical jolts). Orange line: Acceleration magnitude. Cyan line: Gyroscope pitch/roll.',
        'Red/yellow highlighted regions: Manually labeled speed breakers. The tool maps these annotations back to exact timestamps.',
        'Labeled CSV is exported for ground-truth dataset creation.'
    ])

img_path = os.path.join(IMG_DIR, 'labeling_data.png')
if os.path.exists(img_path):
    s.shapes.add_picture(img_path, Inches(0.3), Inches(3.5), width=Inches(9.2))

# ========== SLIDE 13: Fig 4 — Map Visualization ==========
s = slides[12]
shape = find_shape(s, 'Shape;149')
if shape:
    clear_and_set(shape, ['Fig 4: Interactive Road Health Map'])

shape = find_shape(s, 'Shape;148')
if shape:
    clear_and_set(shape, [
        'Detected events plotted on a Leaflet.js map of Bangalore.',
        'Road segments are color-coded: Green = Smooth, Yellow = Degraded, Red = Terrible.',
        'Right panel: List of all detected events (6 potholes, 11 speed breakers, 1 rough patch) with their acceleration and speed values.',
        'Bottom panel: Filtered acceleration and gyroscope waveform for the full ride with anomaly markers.'
    ])

img_path = os.path.join(IMG_DIR, 'map_view.png')
if os.path.exists(img_path):
    s.shapes.add_picture(img_path, Inches(0.3), Inches(3.5), width=Inches(9.2))

# ========== SLIDE 14: Future Work ==========
s = slides[13]
shape = find_shape(s, 'Shape;156')
if shape:
    clear_and_set(shape, ['Future Work'])

shape = find_shape(s, 'Shape;155')
if shape:
    clear_and_set(shape, [
        'Develop a machine learning model trained on the labeled datasets to automatically classify road anomalies.',
        'Build a vehicle speed calibration profile so that detection thresholds adapt to different speeds.',
        'Integrate camera-based visual detection to cross-verify sensor data (multi-sensor fusion).',
        'Add auto-spike detection to the labeling tool to reduce manual annotation effort.',
        'Deploy a public-facing road health dashboard for municipal authorities.'
    ])

# Clear the IEEE format example text from shape 158
shape = find_shape(s, 'Shape;158')
if shape:
    clear_and_set(shape, [''])

# ========== SLIDE 15: References ==========
s = slides[14]
shape = find_shape(s, 'Shape;164')
if shape:
    clear_and_set(shape, ['References'])

shape = find_shape(s, 'Shape;163')
if shape:
    clear_and_set(shape, [
        '[1] J. Eriksson et al., "The Pothole Patrol: Using a Mobile Sensor Network for Road Surface Monitoring," ACM MobiSys, 2008.',
        '[2] A. Mednis et al., "Real-time pothole detection using Android smartphones with accelerometers," IEEE DCOSS, 2011.',
        '[3] M. Perttunen et al., "Distributed road surface condition monitoring using mobile phones," IEEE Pervasive, 2011.',
        '[4] Y.-C. Tai et al., "Automatic road anomaly detection using smart mobile device," IEEE WASA, 2010.',
        '[5] Chart.js — Open Source HTML5 Charts. https://www.chartjs.org/',
        '[6] Papa Parse — CSV Parser for JavaScript. https://www.papaparse.com/',
        '[7] Leaflet.js — Interactive Maps. https://leafletjs.com/'
    ])

shape = find_shape(s, 'Shape;166')
if shape:
    clear_and_set(shape, [''])

# ========== SLIDE 16: Thank You (keep as-is) ==========

prs.save(output_path)
print(f"Presentation saved: {output_path}")
