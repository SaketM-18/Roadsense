from pptx import Presentation
from pptx.util import Inches, Pt

prs = Presentation()

# Slide 1: Title
slide = prs.slides.add_slide(prs.slide_layouts[0])
title = slide.shapes.title
subtitle = slide.placeholders[1]
title.text = "RoadSense: Week 1 to 4 Progress"
subtitle.text = "Building the Foundation & Custom Labeling Tools\nSaket M"

# Slide 2: Introduction
slide = prs.slides.add_slide(prs.slide_layouts[1])
title = slide.shapes.title
title.text = "Week 1 to 4: The Goal"
body = slide.placeholders[1]
tf = body.text_frame
tf.text = "Objective: Establish a robust pipeline for collecting and labeling raw sensor data."
p = tf.add_paragraph()
p.text = "Challenges faced:"
p.level = 1
p = tf.add_paragraph()
p.text = "Raw accelerometer and gyroscope data from smartphones is noisy."
p.level = 2
p = tf.add_paragraph()
p.text = "Existing labeling tools (like CVAT or LabelImg) are designed for images, not 1D time-series sensor data."
p.level = 2
p = tf.add_paragraph()
p.text = "We needed a custom solution to visually map Z-axis jolts to physical potholes."
p.level = 2

# Slide 3: What We Used
slide = prs.slides.add_slide(prs.slide_layouts[1])
title = slide.shapes.title
title.text = "Technologies Used"
body = slide.placeholders[1]
tf = body.text_frame
tf.text = "Frontend Technologies:"
p = tf.add_paragraph()
p.text = "Vanilla HTML5, CSS3, and JavaScript"
p.level = 1
p = tf.add_paragraph()
p.text = "Glassmorphism UI design for a premium, distraction-free dark mode"
p.level = 1
p = tf.add_paragraph()
p.text = "Data Visualization & Processing:"
p.level = 0
p = tf.add_paragraph()
p.text = "Chart.js: For rendering high-performance interactive graphs of the Z-axis acceleration."
p.level = 1
p = tf.add_paragraph()
p.text = "Papa Parse: For rapid, client-side parsing of large CSV files."
p.level = 1

# Slide 4: What We Did (The Labeling Tool)
slide = prs.slides.add_slide(prs.slide_layouts[1])
title = slide.shapes.title
title.text = "What We Did: Custom Sensor Labeling Tool"
body = slide.placeholders[1]
tf = body.text_frame
tf.text = "Developed a web-based application to annotate time-series CSVs."
p = tf.add_paragraph()
p.text = "Interactive Charting: Users can drag and zoom over the acceleration timeline."
p.level = 1
p = tf.add_paragraph()
p.text = "Region Selection: Highlight a section of the graph and label it as 'Pothole' or 'Speed Breaker'."
p.level = 1
p = tf.add_paragraph()
p.text = "Export functionality: The tool maps the labeled regions back to the exact timestamps and exports a new annotated CSV."
p.level = 1
p = tf.add_paragraph()
p.text = "Undo/Redo system and keyboard shortcuts for rapid workflow."
p.level = 1

# Slide 5: Pictures of What We Did
slide = prs.slides.add_slide(prs.slide_layouts[5]) # Title only layout
title = slide.shapes.title
title.text = "Visual Data Labeling (Bounding Boxes)"

img_path = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\main\ml_pipeline\yolo_training\mixed_tuning\val_batch0_labels.jpg'
try:
    # Add image centered
    slide.shapes.add_picture(img_path, Inches(1), Inches(2), width=Inches(8))
except:
    # Fallback if image not found
    txBox = slide.shapes.add_textbox(Inches(2), Inches(3), Inches(6), Inches(1))
    txBox.text_frame.text = "[Please insert screenshot of your labeled data here]"

# Slide 6: Future Improvements
slide = prs.slides.add_slide(prs.slide_layouts[1])
title = slide.shapes.title
title.text = "Future Improvements (Aimed at Week 4)"
body = slide.placeholders[1]
tf = body.text_frame
tf.text = "Tool Enhancements:"
p = tf.add_paragraph()
p.text = "Implement LocalStorage auto-saving to prevent data loss on browser refresh."
p.level = 1
p = tf.add_paragraph()
p.text = "Data decimation for rendering massive 30-minute ride files without lag."
p.level = 1
p = tf.add_paragraph()
p.text = "Add 'Auto-Spike Detection' to automatically highlight regions where Z-axis > 12 m/s²."
p.level = 1
p = tf.add_paragraph()
p.text = "Project Evolution:"
p.level = 0
p = tf.add_paragraph()
p.text = "Transition from sensor-only detection to Multi-Sensor Fusion (Camera + YOLOv8 + Accelerometer)."
p.level = 1

prs.save(r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\Review1_Presentation.pptx')
print("Presentation updated successfully with pictures!")
