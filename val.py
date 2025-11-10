from ultralytics import YOLO

model = YOLO("weights/UIIS-Depth.pt")

results = model.val(
    data="UIIS-Depth.yaml",  
    device='0',  
    workers=8,  
    batch=8,  
)
