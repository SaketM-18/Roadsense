"""
Simply remove all page numbers from the footer of every section.
"""
from docx import Document

doc_path = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\main\RoadSense_Internship_Report_Final.docx'
out_path = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\main\RoadSense_Report_Clean.docx'

doc = Document(doc_path)

for section in doc.sections:
    footer = section.footer
    footer.is_linked_to_previous = False
    for p in footer.paragraphs:
        p.clear()

doc.save(out_path)
print(f"Done! Page numbers removed. Saved to {out_path}")
