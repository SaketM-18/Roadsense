from pptx import Presentation
from pptx.util import Inches

template_path = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\PPT Format_SummerInternshp_Review2 (2).pptx'
prs = Presentation(template_path)

# Remove all existing slides from the template so we start fresh, keeping only the masters
xml_slides = prs.slides._sldIdLst  
slides = list(xml_slides)
for slide in slides:
    xml_slides.remove(slide)

# Slide 1: Title
try:
    slide_layout_title = prs.slide_layouts[0]
except:
    slide_layout_title = prs.slide_layouts[1]
slide = prs.slides.add_slide(slide_layout_title)
if slide.shapes.title:
    slide.shapes.title.text = "RoadSense: Week 1 to 4 Progress"
if len(slide.placeholders) > 1:
    slide.placeholders[1].text = "Data Labeling Foundation\nSaket M"

# Slide 2: Introduction
slide_layout_body = prs.slide_layouts[1]
slide = prs.slides.add_slide(slide_layout_body)
if slide.shapes.title:
    slide.shapes.title.text = "Week 1 to 4: The Goal"
if len(slide.placeholders) > 1:
    tf = slide.placeholders[1].text_frame
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
slide = prs.slides.add_slide(slide_layout_body)
if slide.shapes.title:
    slide.shapes.title.text = "Technologies Used"
if len(slide.placeholders) > 1:
    tf = slide.placeholders[1].text_frame
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

# Slide 4: What We Did
slide = prs.slides.add_slide(slide_layout_body)
if slide.shapes.title:
    slide.shapes.title.text = "What We Did: Custom Sensor Labeling Tool"
if len(slide.placeholders) > 1:
    tf = slide.placeholders[1].text_frame
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

# Slide 5: Pictures
try:
    slide_layout_pic = prs.slide_layouts[5] # Title Only
except:
    slide_layout_pic = slide_layout_body
slide = prs.slides.add_slide(slide_layout_pic)
if slide.shapes.title:
    slide.shapes.title.text = "Visual Data Labeling (Bounding Boxes)"
img_path = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\main\ml_pipeline\yolo_training\mixed_tuning\val_batch0_labels.jpg'
try:
    slide.shapes.add_picture(img_path, Inches(1), Inches(2), width=Inches(8))
except Exception as e:
    print(f"Image error: {e}")

# Slide 6: Future Improvements
slide = prs.slides.add_slide(slide_layout_body)
if slide.shapes.title:
    slide.shapes.title.text = "Future Improvements (Aimed at Week 4)"
if len(slide.placeholders) > 1:
    tf = slide.placeholders[1].text_frame
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

out_path = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\Review1_Presentation_Template.pptx'
prs.save(out_path)
print("Saved template presentation!")
