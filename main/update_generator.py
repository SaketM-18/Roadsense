import os

file_path = r'd:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\main\generate_report.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace titles
content = content.replace('ROADSENSE: AI-POWERED ROAD INFRASTRUCTURE INTELLIGENCE USING MULTI-SENSOR FUSION', 'ROADSENSE: ROAD INFRASTRUCTURE MONITORING USING MULTI-SENSOR FUSION')
content = content.replace('RoadSense: AI-Powered Road Infrastructure Intelligence Using Multi-Sensor Fusion', 'RoadSense: Road Infrastructure Monitoring Using Multi-Sensor Fusion')

# Replace names
content = content.replace('SAKET MEDA', 'SAKET M')
content = content.replace('Saket Meda', 'Saket M')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated generate_report.py with correct title and name.")
