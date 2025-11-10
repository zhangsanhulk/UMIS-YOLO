from ultralytics import YOLO

def main():
    model = YOLO("UMIS-YOLOv8m.yaml")
    train_results = model.train(
    data="UIIS-Depth.yaml",  # path to dataset YAML
    epochs=300,  # number of training epochs
    device='0',  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    workers=8,
    batch=8,
    amp=False,
    patience=50
    )


if __name__ == '__main__':
    main()

