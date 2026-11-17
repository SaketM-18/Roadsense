"""
Fix page numbering in the Final report:
- Preliminary pages (cover -> list of acronyms): no page numbers
- Main content (Chapter 1 Introduction onwards): page numbers starting from 1
"""
from docx import Document
from docx.oxml import OxmlElement, ns
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm
import copy

doc_path = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\main\RoadSense_Internship_Report_Final.docx'

doc = Document(doc_path)

# Step 1: Find the paragraph for "CHAPTER 1" (the actual heading, not the TOC entry)
# From our scan: paragraph 147 is "CHAPTER 1" in the main body
chapter1_idx = None
for i, p in enumerate(doc.paragraphs):
    if i > 130 and p.text.strip() == 'CHAPTER 1':
        chapter1_idx = i
        break

if chapter1_idx is None:
    print("ERROR: Could not find CHAPTER 1 heading!")
    exit(1)

print(f"Found CHAPTER 1 at paragraph index {chapter1_idx}")

# Step 2: Insert a section break (new page) right before CHAPTER 1.
# This is done by adding a sectPr to the paragraph BEFORE chapter 1,
# which tells Word "end the current section here and start a new one".

# Get the XML element of the paragraph just before Chapter 1
prev_para = doc.paragraphs[chapter1_idx - 1]
prev_element = prev_para._element

# Create a section properties element for the preliminary section
sectPr = OxmlElement('w:sectPr')

# Set page size and margins to match the existing section
existing_section = doc.sections[0]
pgSz = OxmlElement('w:pgSz')
pgSz.set(ns.qn('w:w'), str(existing_section.page_width))
pgSz.set(ns.qn('w:h'), str(existing_section.page_height))
sectPr.append(pgSz)

pgMar = OxmlElement('w:pgMar')
pgMar.set(ns.qn('w:top'), str(existing_section.top_margin))
pgMar.set(ns.qn('w:bottom'), str(existing_section.bottom_margin))
pgMar.set(ns.qn('w:left'), str(existing_section.left_margin))
pgMar.set(ns.qn('w:right'), str(existing_section.right_margin))
sectPr.append(pgMar)

# Set section break type to "next page"
sectType = OxmlElement('w:type')
sectType.set(ns.qn('w:val'), 'nextPage')
sectPr.append(sectType)

# Append the sectPr to the last paragraph of the preliminary section
# This is done by adding it as a child of the paragraph's pPr
pPr = prev_element.find(ns.qn('w:pPr'))
if pPr is None:
    pPr = OxmlElement('w:pPr')
    prev_element.insert(0, pPr)
pPr.append(sectPr)

# Step 3: Now the document has 2 sections.
# Reload to pick up the new sections
out_path = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\main\RoadSense_Report_PageFixed.docx'
doc.save(out_path)
doc = Document(out_path)

print(f"Document now has {len(doc.sections)} sections")

# Step 4: For section 0 (preliminary), remove/clear footer (no page numbers)
section0 = doc.sections[0]
footer0 = section0.footer
footer0.is_linked_to_previous = False
for p in footer0.paragraphs:
    p.text = ''
    p.clear()

# Step 5: For section 1 (main content), add page numbers starting from 1
section1 = doc.sections[1]
footer1 = section1.footer
footer1.is_linked_to_previous = False

# Clear existing footer content
for p in footer1.paragraphs:
    p.text = ''
    p.clear()

if not footer1.paragraphs:
    p = footer1.add_paragraph()
else:
    p = footer1.paragraphs[0]

p.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Add the PAGE field
run1 = p.add_run()
fldChar1 = OxmlElement('w:fldChar')
fldChar1.set(ns.qn('w:fldCharType'), 'begin')
run1._r.append(fldChar1)

run2 = p.add_run()
instrText = OxmlElement('w:instrText')
instrText.set(ns.qn('xml:space'), 'preserve')
instrText.text = "PAGE"
run2._r.append(instrText)

run3 = p.add_run()
fldChar2 = OxmlElement('w:fldChar')
fldChar2.set(ns.qn('w:fldCharType'), 'separate')
run3._r.append(fldChar2)

run4 = p.add_run("1")

run5 = p.add_run()
fldChar3 = OxmlElement('w:fldChar')
fldChar3.set(ns.qn('w:fldCharType'), 'end')
run5._r.append(fldChar3)

# Step 6: Set page numbering to restart from 1 in section 1
# This is done via the sectPr's pgNumType element
sectPr1 = section1._sectPr
pgNumType = sectPr1.find(ns.qn('w:pgNumType'))
if pgNumType is None:
    pgNumType = OxmlElement('w:pgNumType')
    sectPr1.append(pgNumType)
pgNumType.set(ns.qn('w:start'), '1')

doc.save(out_path)
print("Done! Page numbers now start from 1 at Chapter 1 Introduction.")
print("Preliminary pages (cover, certificate, abstract, etc.) have no page numbers.")
