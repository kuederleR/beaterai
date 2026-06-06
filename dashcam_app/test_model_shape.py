import torch
from yolop_detector import YolopDetector

detector = YolopDetector()
img = torch.zeros(1, 3, 416, 640).to(detector.device)
if detector.half:
    img = img.half()
img = img.type_as(next(detector.model.parameters()))

try:
    with torch.no_grad():
        out = detector.model(img)
        print("Success with 416x640")
        print("Shapes:", [o.shape if isinstance(o, torch.Tensor) else "tuple" for o in out])
except Exception as e:
    print("Failed with 416x640:", e)

