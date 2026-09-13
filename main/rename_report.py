"""
Script to rename the project title in the Word document.
"""
from docx import Document
import os

doc_path = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\main\RoadSense_Internship_Report.docx'
doc = Document(doc_path)

old_title_1 = "ROADSENSE: AI-POWERED ROAD INFRASTRUCTURE INTELLIGENCE\nUSING MULTI-SENSOR FUSION"
new_title_1 = "ROADSENSE: ROAD INFRASTRUCTURE MONITORING\nUSING MULTI-SENSOR FUSION"

old_title_2 = "ROADSENSE: AI-POWERED ROAD INFRASTRUCTURE INTELLIGENCE USING MULTI-SENSOR FUSION"
new_title_2 = "ROADSENSE: ROAD INFRASTRUCTURE MONITORING USING MULTI-SENSOR FUSION"

old_title_3 = "RoadSense: AI-Powered Road Infrastructure Intelligence Using Multi-Sensor Fusion"
new_title_3 = "RoadSense: Road Infrastructure Monitoring Using Multi-Sensor Fusion"

def replace_text_in_runs(paragraph, old_text, new_text):
    # This simple replacement works if the text is fully contained in a single run.
    # Because python-docx splits text into runs unpredictably, the safest way for exact matches
    # that might span runs is to clear runs and rewrite, but we can first check if paragraph.text contains it.
    if old_text in paragraph.text:
        # To preserve formatting, if it's the exact paragraph text, we just replace in the first run and clear others
        if paragraph.text.strip() == old_text.strip():
            if paragraph.runs:
                # save formatting of first run
                r0 = paragraph.runs[0]
                font_name = r0.font.name
                font_size = r0.font.size
                font_bold = r0.bold
                font_italic = r0.italic
                
                for r in paragraph.runs:
                    r.text = ''
                
                paragraph.runs[0].text = new_text
                paragraph.runs[0].font.name = font_name
                paragraph.runs[0].font.size = font_size
                paragraph.runs[0].bold = font_bold
                paragraph.runs[0].italic = font_italic
        else:
            # If it's embedded in other text, we can try replacing run by run (risky) or just reconstruct it.
            # Usually titles are their own paragraphs.
            text = paragraph.text.replace(old_text, new_text)
            if paragraph.runs:
                r0 = paragraph.runs[0]
                font_name = r0.font.name
                font_size = r0.font.size
                font_bold = r0.bold
                font_italic = r0.italic
                
                for r in paragraph.runs:
                    r.text = ''
                
                paragraph.runs[0].text = text
                paragraph.runs[0].font.name = font_name
                paragraph.runs[0].font.size = font_size
                paragraph.runs[0].bold = font_bold
                paragraph.runs[0].italic = font_italic

for para in doc.paragraphs:
    replace_text_in_runs(para, old_title_1, new_title_1)
    replace_text_in_runs(para, old_title_2, new_title_2)
    replace_text_in_runs(para, old_title_3, new_title_3)
    
    # Also just in case it was written without 'ROADSENSE: '
    replace_text_in_runs(para, "AI-POWERED ROAD INFRASTRUCTURE INTELLIGENCE\nUSING MULTI-SENSOR FUSION", "ROAD INFRASTRUCTURE MONITORING\nUSING MULTI-SENSOR FUSION")
    replace_text_in_runs(para, "AI-POWERED ROAD INFRASTRUCTURE INTELLIGENCE USING MULTI-SENSOR FUSION", "ROAD INFRASTRUCTURE MONITORING USING MULTI-SENSOR FUSION")
    replace_text_in_runs(para, "AI-Powered Road Infrastructure Intelligence Using Multi-Sensor Fusion", "Road Infrastructure Monitoring Using Multi-Sensor Fusion")

# Also check tables (like the title page table if any)
for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            for para in cell.paragraphs:
                replace_text_in_runs(para, old_title_1, new_title_1)
                replace_text_in_runs(para, old_title_2, new_title_2)
                replace_text_in_runs(para, old_title_3, new_title_3)
                replace_text_in_runs(para, "AI-POWERED ROAD INFRASTRUCTURE INTELLIGENCE\nUSING MULTI-SENSOR FUSION", "ROAD INFRASTRUCTURE MONITORING\nUSING MULTI-SENSOR FUSION")
                replace_text_in_runs(para, "AI-POWERED ROAD INFRASTRUCTURE INTELLIGENCE USING MULTI-SENSOR FUSION", "ROAD INFRASTRUCTURE MONITORING USING MULTI-SENSOR FUSION")
                replace_text_in_runs(para, "AI-Powered Road Infrastructure Intelligence Using Multi-Sensor Fusion", "Road Infrastructure Monitoring Using Multi-Sensor Fusion")


doc.save(doc_path)
print("Title updated successfully in the report.")
