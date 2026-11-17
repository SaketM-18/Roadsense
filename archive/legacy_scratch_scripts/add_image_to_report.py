"""
Insert the RoadSense app screenshot into Chapter 4 Results section,
right after section 4.3 Analysis (paragraph 244) and before 4.4 Comparison (paragraph 245).
"""
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, ns

img_path = r'C:\Users\medas\.gemini\antigravity\brain\653bea0c-52ca-450a-99c0-1bd76a40aee6\.user_uploaded\media__1784829977589.png'
doc_path = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\main\RoadSense_Internship_Report.docx'

doc = Document(doc_path)

# Find the ACTUAL 4.4 heading (not the TOC one) by looking for it after index 230+
target_idx = None
for i, p in enumerate(doc.paragraphs):
    if i > 230 and '4.4' in p.text and 'COMPARISON' in p.text.upper():
        target_idx = i
        break

if target_idx is None:
    print("ERROR: Could not find the actual 4.4 section heading!")
    exit(1)

print(f"Found actual 4.4 heading at paragraph index {target_idx}")
print(f"Text: {doc.paragraphs[target_idx].text[:80]}")

# We need to insert BEFORE paragraph at target_idx.
# python-docx doesn't have a direct "insert paragraph at index" method,
# so we manipulate the XML directly.

target_para = doc.paragraphs[target_idx]
target_element = target_para._element

# Create empty spacer paragraph
spacer1 = OxmlElement('w:p')
target_element.addprevious(spacer1)

# Create the image paragraph
img_para = doc.add_paragraph()  # temporary, we'll move it
img_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = img_para.add_run()
run.add_picture(img_path, width=Inches(5.5))

# Move the image paragraph element before the target
img_element = img_para._element
target_element.addprevious(img_element)

# Create the caption paragraph
caption_para = doc.add_paragraph()
caption_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap_run = caption_para.add_run('Figure 4.2: RoadSense Web App — Dark-Mode Map with Colour-Coded Road Segments')
cap_run.italic = True
cap_run.font.size = Pt(11)
cap_run.font.name = 'Times New Roman'

# Move caption before the target too (after the image)
caption_element = caption_para._element
target_element.addprevious(caption_element)

# Add another spacer after caption
spacer2 = OxmlElement('w:p')
target_element.addprevious(spacer2)

# Now add page numbers to footer
def add_page_number_to_footer(doc):
    for section in doc.sections:
        footer = section.footer
        for p in footer.paragraphs:
            p.text = ''
            p.clear()
        if not footer.paragraphs:
            p = footer.add_paragraph()
        else:
            p = footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Field start
        run1 = p.add_run()
        fldChar1 = OxmlElement('w:fldChar')
        fldChar1.set(ns.qn('w:fldCharType'), 'begin')
        run1._r.append(fldChar1)
        
        # Instruction
        run2 = p.add_run()
        instrText = OxmlElement('w:instrText')
        instrText.set(ns.qn('xml:space'), 'preserve')
        instrText.text = "PAGE"
        run2._r.append(instrText)
        
        # Separator
        run3 = p.add_run()
        fldChar2 = OxmlElement('w:fldChar')
        fldChar2.set(ns.qn('w:fldCharType'), 'separate')
        run3._r.append(fldChar2)
        
        # Placeholder
        run4 = p.add_run("1")
        
        # Field end
        run5 = p.add_run()
        fldChar3 = OxmlElement('w:fldChar')
        fldChar3.set(ns.qn('w:fldCharType'), 'end')
        run5._r.append(fldChar3)

add_page_number_to_footer(doc)

doc.save(doc_path)
print("Done! Image inserted in the correct location and page numbers added.")
