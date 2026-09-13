from docx import Document

doc_path = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\main\RoadSense_Report_Final_Paged.docx'
out_path = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\main\RoadSense_Report_Final_Sync.docx'

doc = Document(doc_path)

replacements = {
    '1.1 Introduction .................................................... 1': '1.1 Introduction .................................................... 1',
    '1.2 Aim and Objectives ......................................... 3': '1.2 Aim and Objectives ......................................... 2',
    '1.3 Problem Statement .......................................... 4': '1.3 Problem Statement .......................................... 3',
    '1.4 Scope of the Project ........................................ 5': '1.4 Scope of the Project ........................................ 3',
    '1.5 Challenges ..................................................... 6': '1.5 Challenges ..................................................... 3',
    '2.1 Literature Survey ............................................. 8': '2.1 Literature Survey ............................................. 4',
    '2.2 Requirements .................................................. 12': '2.2 Requirements .................................................. 5',
    '3.1 Proposed Methodology ..................................... 14': '3.1 Proposed Methodology ..................................... 6',
    '3.2 Datasets .......................................................... 17': '3.2 Datasets .......................................................... 7',
    '3.3 Flow of the Model ........................................... 19': '3.3 Flow of the Model ........................................... 8',
    '4.1 Experimental Setup ......................................... 22': '4.1 Experimental Setup ......................................... 9',
    '4.2 Result Summary Tables .................................... 24': '4.2 Result Summary Tables .................................... 9',
    '4.3 Analysis of Results .......................................... 25': '4.3 Analysis of Results .......................................... 10',
    '4.4 Comparison with Existing Methods .................. 27': '4.4 Comparison with Existing Methods .................. 11',
    '5.1 Key Findings ................................................... 29': '5.1 Key Findings ................................................... 12',
    '5.2 Interpretation of Results .................................. 30': '5.2 Interpretation of Results .................................. 12',
    '5.3 Implications of the Study ................................. 31': '5.3 Implications of the Study ................................. 13',
    '5.4 Fulfillment of Objectives ................................. 32': '5.4 Fulfillment of Objectives ................................. 13',
    '6.1 Results ............................................................ 34': '6.1 Results ............................................................ 14',
    '6.2 Future Work .................................................... 35': '6.2 Future Work .................................................... 14',
    '6.3 References ...................................................... 37': '6.3 References ...................................................... 15',
    'APPENDIX ........................................................... 38': 'APPENDIX ........................................................... 16',
}

def replace_in_paragraph(p):
    for old, new in replacements.items():
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
            return # Only replace once per paragraph

for p in doc.paragraphs:
    replace_in_paragraph(p)

doc.save(out_path)
print("TOC synced successfully!")
