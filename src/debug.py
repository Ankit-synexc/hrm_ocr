import cv2
import numpy as np
from hrm_ocr.models.ocr_engine import get_engine
from hrm_ocr.models.template_detector import detect_template

engine = get_engine("en")

# Create a blank image and put some text to simulate an aadhaar card
img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
cv2.putText(img, "Government of India", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
cv2.putText(img, "Ankit Choudhary", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
cv2.putText(img, "DOB: 28/06/2006", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
cv2.putText(img, "Male", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
cv2.putText(img, "8852 0053 4711", (100, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

full_regions = engine.recognize_full_card(img)
full_text = " ".join([r.text for r in full_regions])
print("Full text:", full_text)

template = detect_template(img, extracted_text=full_text)
print("Template:", template.doc_type, template.template_version)
print("Coords map keys:", list(template.field_coordinate_map.keys()))

spatial = engine.recognize_spatial_from_regions(full_regions, template.doc_type)
print("Spatial results:", list(spatial.keys()))

if not spatial or "aadhaar_number" not in spatial:
    print("Falling back to coordinate map!")
    fallback = engine.recognize_all_fields(img, template.field_coordinate_map)
    print("Fallback results:", list(fallback.keys()))
