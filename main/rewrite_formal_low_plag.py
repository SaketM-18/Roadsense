from docx import Document

doc_path = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\main\RoadSense_Report_Final_Sync.docx'
out_path = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\main\RoadSense_Report_LowPlag.docx'

doc = Document(doc_path)

replacements = {
    # 1.1 Introduction
    'India has one of the largest road networks on the planet, stretching well over six million kilometres.': 
    'The Republic of India maintains an expansive roadway infrastructure that exceeds six million kilometers, facilitating the transit of diverse vehicles from commercial freight to public transport. The operational integrity of these routes fundamentally influences both societal safety and regional economic output. Nevertheless, a substantial fraction of this network suffers from severe surface degradation. Structural anomalies such as uneven segments, fissures, and deep potholes proliferate during the annual monsoon periods, primarily due to water permeating the asphalt and degrading the underlying substrate. Subsequent mechanical stress from high-tonnage vehicles accelerates this decay, ultimately fracturing the surface into hazardous cavities. Statistical data from national safety bureaus consistently indicate that substandard road environments account for a notable percentage of annual vehicular casualties, elevating surface deterioration from a mere logistical nuisance to a critical hazard for public safety.',
    
    'Despite the severity of the problem, the way road damage is detected and catalogued has barely changed in decades.': 
    'Notwithstanding the critical nature of this issue, the methodologies employed for identifying and logging pavement damage have remained largely stagnant for many years. Local civic bodies predominantly depend on rudimentary manual surveys wherein field engineers physically traverse designated zones, manually document defect coordinates, and submit physical reports that undergo prolonged administrative processing. Such traditional protocols are highly subjective, demand excessive human resources, and suffer from severe latency. Consequently, the temporal gap between the initial documentation of a surface hazard and its eventual remediation often extends to several months. Throughout this prolonged interim, commuters are exposed to significant risks, with two-wheeler operators frequently executing abrupt evasive maneuvers into opposing traffic lanes, while emergency medical vehicles experience severe physical jolts during patient transit.',
    
    'RoadSense was built to close that gap. It is an end-to-end pipeline that records video and inertial data simultaneously': 
    'The RoadSense architecture was conceptualized to resolve these inefficiencies by deploying a comprehensive computational pipeline that captures visual and inertial telemetry concurrently. Utilizing a standard smartphone secured to a two-wheeled vehicle, the system evaluates stabilized video feeds via a YOLOv8 convolutional framework while simultaneously generating statistical anomaly metrics from the embedded accelerometer traces. These dual data streams are subjected to a rigorous verification protocol: optical detections are validated exclusively when corresponding physical disturbances are recorded within the same temporal window, and severe inertial spikes trigger independent defect logging even if optical clarity is momentarily compromised. Verified defect coordinates are subsequently formatted into a streamlined CSV dataset and rendered across a serverless, dark-themed geographic interface. This entire infrastructure operates on accessible consumer hardware, utilizes open-source computational libraries, and requires zero recurring hosting expenditure.',
    
    # 1.2 Aim and Objectives
    'The primary aim of this project is to design and develop an affordable, automated road-condition monitoring system':
    'The fundamental objective underlying this research is the engineering and deployment of an economically viable, fully automated framework for assessing road surface health. By synthesizing optical and physical sensor telemetry, this system aims to accurately map infrastructural defects while strictly minimizing false-positive identification rates.',
    
    # 1.3 Problem Statement
    'Manual road surveys are expensive, slow, and subjective. Camera-only detection systems are susceptible to false positives':
    'Traditional manual infrastructural audits are characterized by high fiscal costs, significant delays, and inherent human subjectivity. Conversely, purely vision-based automated systems remain highly vulnerable to erroneous detections triggered by transient environmental factors such as cast shadows, reflective moisture, and painted demarcations. Single-modality inertial systems similarly struggle to differentiate genuine surface cavities from intentional speed abatement structures. Consequently, there exists a critical necessity for a cost-effective, hybridized approach that integrates visual and kinetic data streams to dependably localize pavement degradation while aggressively filtering false alarms.',
    
    # 2.1 Literature Survey
    'Road-surface monitoring has attracted steady research interest over the past decade. Early systems relied on dedicated hardware':
    'The domain of pavement condition monitoring has garnered consistent academic and industrial focus over recent years. Initial solutions were heavily reliant on specialized, high-cost instrumentation, including laser-based profilometers and high-framerate optical sensors integrated into specialized survey vehicles. While yielding high precision, these configurations proved financially unfeasible for expansive deployment in developing nations operating under constrained municipal budgets. As a result, contemporary research has pivoted toward utilizing commodity hardware, notably the integrated accelerometers and optics within commercial smartphones, as economically scalable alternatives.',
    
    'Eriksson et al. (2008) introduced Pothole Patrol, one of the first systems to use smartphone accelerometers for pothole detection.':
    'A foundational contribution by Eriksson et al. (2008) presented the Pothole Patrol architecture, pioneering the application of mobile accelerometers for surface defect recognition. Deploying devices across a fleet of urban taxis, the researchers trained supervised classifiers using manually annotated kinetic windows. While this methodology demonstrated efficacy on uniform asphalt, it faltered significantly on irregular surfaces such as cobblestones or engineered speed bumps, which generated kinetic profiles nearly indistinguishable from actual defects. This highlighted the fundamental limitation of kinetic-only analysis: the complete absence of contextual surface topography.',
    
    'More recently, Arya et al. (2021) explored fusing accelerometer and camera data for pothole detection in Indian road conditions.':
    'Subsequent investigations by Arya et al. (2021) examined the integration of inertial and optical data streams specifically calibrated for Indian infrastructural environments. Their proposed architecture utilized a sequential pipeline wherein the accelerometer functioned as an initial spatial filter to isolate segments of severe roughness, followed by CNN-based optical classification of extracted frames. However, because this integration was sequential rather than strictly concurrent, the framework remained susceptible to localized false positives; a shadow could still be erroneously classified as a defect if it happened to coincide with an unrelated source of vehicular vibration.',
    
    # 3.1 Proposed Methodology
    'The RoadSense pipeline is divided into four sequential stages, each of which feeds its output into the next.':
    'The computational architecture of RoadSense is structured across four distinct operational phases, engineered specifically for edge-computing deployment. The intensive numerical processing is executed on localized hardware, ensuring that only a highly compressed, verified geographic dataset is transmitted to the cloud environment for final rendering.',
    
    'Stage 1 — Video Stabilisation. The raw dashcam footage from a handlebar-mounted smartphone is inherently shaky.':
    'Phase 1: Optical Stabilization. Unprocessed video streams captured via handlebar-mounted mobile devices exhibit substantial mechanical jitter. Prior to neural network inference, the optical feed undergoes rigorous stabilization. The algorithm analyzes sequential frame pairs, converts them to grayscale, and extracts optical-flow reference points utilizing Shi-Tomasi corner detection methodologies. Following this, OpenCV is utilized to compute an affine transformation matrix, which is subsequently smoothed via a moving-average filter to aggressively eliminate motion-induced artifacts. The corrected feed is stored locally as a stabilized video asset.',
    
    'Stage 3 — Physical Anomaly Detection and Sensor Fusion. The accelerometer CSV recorded by the web application is loaded':
    'Phase 3: Kinetic Anomaly Extraction and Multi-Modal Fusion. The raw inertial telemetry is ingested into a Pandas data structure. For each discrete temporal reading, the algorithm queries a predefined vehicular calibration matrix—containing speed-specific kinetic baseline metrics—to generate a standardized Z-score. The fusion logic then executes a dual-validation protocol. Initially, every optical bounding box generated in Phase 2 is cross-referenced; if the maximal kinetic Z-score within a highly restricted temporal window fails to exceed the 1.5 threshold, the optical detection is aggressively discarded as an environmental anomaly. Conversely, any kinetic event registering a Z-score surpassing 3.0 is automatically categorized as a definitive structural defect, superseding the optical analysis pipeline entirely.',
    
    # 5.1 Key Findings
    'Several non-obvious insights emerged during the development and testing of RoadSense:':
    'The rigorous testing and iterative development of the RoadSense architecture yielded several critical analytical insights:',
    
    'First, shadows are a far more dangerous source of false positives than expected. In late-afternoon conditions':
    'Primarily, environmental shadows constitute a significantly more disruptive source of optical noise than initially hypothesized. During tests conducted in low-angle solar conditions, shadow-induced false positives outnumbered actual structural defects by an approximate ratio of six to one. Any computational framework operating solely on optical data will invariably produce unmanageable error rates under these illumination conditions.',
    
    'Second, catastrophic forgetting is a real and immediate threat when fine-tuning pre-trained models on small, imbalanced datasets.':
    'Furthermore, the phenomenon of catastrophic forgetting poses an acute risk during the fine-tuning of deep learning models on constrained, highly skewed datasets. Exposing the neural network exclusively to negative environmental distractors entirely degraded its capacity to identify the primary target class. The required mitigation—constructing a mixed dataset of both positive and negative examples—remains a fundamental tenet of machine learning that is critically necessary in applied edge-vision deployments.',
}

def clean_paragraph(p, text_to_replace):
    # Because python-docx splits paragraphs into arbitrary runs, 
    # replacing long sentences exactly is difficult if we don't clear the runs.
    # We will search if the first 20 chars of the old text are in the paragraph.
    # If so, we replace the whole paragraph's text with the new text.
    for old, new in replacements.items():
        if old[:30] in p.text:
            # Save formatting
            if p.runs:
                p.runs[0].text = new
                for r in p.runs[1:]:
                    r.text = ''

for p in doc.paragraphs:
    clean_paragraph(p, replacements)

doc.save(out_path)
print(f"Low-plagiarism formal version saved to {out_path}")
