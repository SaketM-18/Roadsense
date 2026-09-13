from docx import Document
from docx.oxml import OxmlElement, ns
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc_path = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\main\RoadSense_Report_Clean.docx'
out_path = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\main\RoadSense_Report_Final_Paged.docx'

doc = Document(doc_path)

chapter1_idx = None
for i, p in enumerate(doc.paragraphs):
    if i > 130 and p.text.strip() == 'CHAPTER 1':
        chapter1_idx = i
        break

prev_para = doc.paragraphs[chapter1_idx - 1]
prev_element = prev_para._element
sectPr = OxmlElement('w:sectPr')
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
sectType = OxmlElement('w:type')
sectType.set(ns.qn('w:val'), 'nextPage')
sectPr.append(sectType)
pPr = prev_element.find(ns.qn('w:pPr'))
if pPr is None:
    pPr = OxmlElement('w:pPr')
    prev_element.insert(0, pPr)
pPr.append(sectPr)

doc.save(out_path)
doc = Document(out_path)

section0 = doc.sections[0]
footer0 = section0.footer
footer0.is_linked_to_previous = False
for p in footer0.paragraphs:
    p.text = ''
    p.clear()

section1 = doc.sections[1]
footer1 = section1.footer
footer1.is_linked_to_previous = False
for p in footer1.paragraphs:
    p.text = ''
    p.clear()

if not footer1.paragraphs:
    p = footer1.add_paragraph()
else:
    p = footer1.paragraphs[0]
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
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

sectPr1 = section1._sectPr
pgNumType = sectPr1.find(ns.qn('w:pgNumType'))
if pgNumType is None:
    pgNumType = OxmlElement('w:pgNumType')
    sectPr1.append(pgNumType)
pgNumType.set(ns.qn('w:start'), '1')

doc.save(out_path)
print("Applied page numbers!")
