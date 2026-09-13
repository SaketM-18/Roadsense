from pptx import Presentation

template_path = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\PPT Format_SummerInternshp_Review2 (2).pptx'
prs = Presentation(template_path)

for i, slide in enumerate(prs.slides):
    print(f"\n--- Slide {i+1} (layout: '{slide.slide_layout.name}') ---")
    for shape in slide.shapes:
        if hasattr(shape, "text") and shape.text.strip():
            text_preview = shape.text[:300].replace('\n', '\\n')
            print(f"  [{shape.name}] '{text_preview}'")
