import os
from docx import Document
from docx.oxml import OxmlElement, ns
from docx.enum.text import WD_ALIGN_PARAGRAPH

def create_element(name):
    return OxmlElement(name)

def create_attribute(element, name, value):
    element.set(ns.qn(name), value)

def add_page_number(paragraph):
    # Field start
    run1 = paragraph.add_run()
    fldChar1 = create_element('w:fldChar')
    create_attribute(fldChar1, 'w:fldCharType', 'begin')
    run1._r.append(fldChar1)

    # Instruction text
    run2 = paragraph.add_run()
    instrText = create_element('w:instrText')
    create_attribute(instrText, 'xml:space', 'preserve')
    instrText.text = "PAGE"
    run2._r.append(instrText)

    # Separator
    run3 = paragraph.add_run()
    fldChar2 = create_element('w:fldChar')
    create_attribute(fldChar2, 'w:fldCharType', 'separate')
    run3._r.append(fldChar2)

    # Dummy text (Word updates this automatically)
    run4 = paragraph.add_run("1")

    # Field end
    run5 = paragraph.add_run()
    fldChar3 = create_element('w:fldChar')
    create_attribute(fldChar3, 'w:fldCharType', 'end')
    run5._r.append(fldChar3)

doc_path = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\main\RoadSense_Internship_Report.docx'
output_path = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\main\RoadSense_Internship_Report_Final.docx'

doc = Document(doc_path)

for section in doc.sections:
    # Accessing the footer
    footer = section.footer
    
    # Clear any existing paragraphs
    for p in footer.paragraphs:
        p.text = ''
        p.clear()
        
    if not footer.paragraphs:
        p = footer.add_paragraph()
    else:
        p = footer.paragraphs[0]
        
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_page_number(p)

doc.save(output_path)
print(f"Page numbers correctly injected into footers. Saved to {output_path}")
