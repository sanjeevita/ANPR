import torch
import cv2
import numpy as np
import easyocr

EASY_OCR = easyocr.Reader(['en']) 
OCR_TH = 0.2

def detectx (frame, model):
    frame = [frame]
    results = model(frame)
    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cordinates

def plot_boxes(results, frame,classes):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.55: 
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) 
            text_d = classes[int(labels[i])]
            coords = [x1,y1,x2,y2]
            plate_num = recognize_plate_easyocr(img = frame, coords= coords, reader= EASY_OCR, region_threshold= OCR_TH)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
            cv2.putText(frame, f"{plate_num}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 2)
    return frame

def recognize_plate_easyocr(img, coords,reader,region_threshold):
    xmin, ymin, xmax, ymax = coords
    nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)] 
    ocr_result = reader.readtext(nplate)
    text = filter_text(region=nplate, ocr_result=ocr_result, region_threshold= region_threshold)
    if len(text) ==1:
        text = text[0].upper()
    return text

def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    plate = [] 
    print(ocr_result)
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        if length*height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate

def main(img_path=None, vid_path=None,vid_out = None):

    model =  torch.hub.load('yolov5-master', 'custom', source ='local', path='best.pt',force_reload=True) 
    classes = model.names 

    if img_path != None:
        frame = cv2.imread(img_path) 
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = detectx(frame, model = model)   
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        frame = plot_boxes(results, frame,classes = classes)
        cv2.imwrite("img_out.jpg",frame)

    elif vid_path !=None:
        cap = cv2.VideoCapture(vid_path)
        if vid_out: 
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(vid_out, codec, fps, (width, height))
        frame_no = 1
        cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            if ret  and frame_no %1 == 0:
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                results = detectx(frame, model = model)
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                frame = plot_boxes(results, frame,classes = classes)
                cv2.imshow("vid_out", frame)
                if vid_out:
                    out.write(frame)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                frame_no += 1
        out.release()
        cv2.destroyAllWindows()

# main(vid_path="./test_images/vid_1.mp4",vid_out="vid_1.mp4") ### for custom video
main(img_path="t_1.jpg") 