import cv2
import queue
import time
import threading
q=queue.Queue()

url = "rtsp://admin:1qaz!QAZ@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"

def Receive():
    print("start Receive")
    cap = cv2.VideoCapture(url)
    ret, frame = cap.read()
    q.put(frame)
    while ret:
        ret, frame = cap.read()
        q.put(frame)
 
 
stop_event = threading.Event()

def Display():
    print("Start Displaying")
    while not stop_event.is_set():
        if not q.empty():
            frame = q.get()
            if frame is not None:
                cv2.imshow("frame1", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Keyboard Interrupt")
            stop_event.set()
            break
    cv2.destroyAllWindows()
 
if __name__=='__main__':
    p1=threading.Thread(target=Receive)
    p2 = threading.Thread(target=Display)
    p1.start()
    p2.start()