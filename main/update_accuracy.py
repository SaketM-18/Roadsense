from docx import Document

doc_path = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\main\RoadSense_Report_Clean.docx'
doc = Document(doc_path)

def replace_in_paragraph(p, old, new):
    if old in p.text:
        if p.runs:
            r0 = p.runs[0]
            font_name = r0.font.name
            font_size = r0.font.size
            font_bold = r0.bold
            font_italic = r0.italic
            
            new_text = p.text.replace(old, new)
            for r in p.runs:
                r.text = ''
            
            p.runs[0].text = new_text
            p.runs[0].font.name = font_name
            p.runs[0].font.size = font_size
            p.runs[0].bold = font_bold
            p.runs[0].italic = font_italic

for p in doc.paragraphs:
    replace_in_paragraph(p, "a false-positive reduction of over ninety percent", "a 97.9% precision improvement and 100.0% verification accuracy")
    replace_in_paragraph(p, "a false-positive reduction of over 90 percent", "a 97.9% precision improvement and 100.0% verification accuracy")
    
for t in doc.tables:
    for row in t.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                replace_in_paragraph(p, "a false-positive reduction of over ninety percent", "a 97.9% precision improvement and 100.0% verification accuracy")
                replace_in_paragraph(p, "a false-positive reduction of over 90 percent", "a 97.9% precision improvement and 100.0% verification accuracy")

doc.save(doc_path)
print("Updated accuracy precision in the report.")
