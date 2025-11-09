# # client_webcam.py
# # Python 3.7 compatible
# import asyncio
# import json
# import base64
# import cv2
# import websockets


# async def main():
#     uri = "ws://localhost:8000/ws"
#     print("Connecting to:", uri)
#     async with websockets.connect(uri) as ws:

#         msg = await ws.recv()
#         print("<<", msg)

#         cap = cv2.VideoCapture(0)

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             ok, buf = cv2.imencode(".jpg", frame)
#             if not ok:
#                 continue

#             b64_frame = base64.b64encode(buf.tobytes()).decode("ascii")

#             await ws.send(json.dumps({"type": "frame", "data": b64_frame}))

#             # Non-blocking read of possible prediction
#             try:
#                 resp = await asyncio.wait_for(ws.recv(), timeout=0.01)
#                 # print("<<", resp)
#                 # resp = await ws.recv()
#                 data = json.loads(resp)
#                 if data.get("type") == "prediction":
#                     print("Detected:", data["label"], f"({data['score']*100:.1f}%)")
#             except asyncio.TimeoutError:
#                 pass

#             cv2.imshow("Client Camera", frame)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     asyncio.get_event_loop().run_until_complete(main())
# client_webcam.py
# Python 3.7 compatible
import asyncio
import json
import base64
import cv2
import websockets

SERVER_WS_URL = "ws://localhost:8000/ws"


async def main():
    print(f"Connecting to {SERVER_WS_URL} ...")
    async with websockets.connect(SERVER_WS_URL) as ws:

        # Receive initial ready message
        hello = await ws.recv()
        print("<<", hello)

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 25)  # match your original script

        print("Streaming webcam frames. Press Q to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed")
                break

            # Encode frame as JPEG → Base64
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            frame_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

            # Send frame to backend
            await ws.send(json.dumps({"type": "frame", "data": frame_b64}))

            # Receive latest recognized sign
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.001)
                data = json.loads(msg)

                if data.get("type") == "prediction":
                    label = data["label"]
                    score = data["score"]

                    if label == "none":
                        print("Detected: none")
                    else:
                        print(f"Detected: {label} ({score*100:.2f}%)")

            except asyncio.TimeoutError:
                pass  # Normal: prediction thread may still be running

            # Show live webcam feed locally (optional)
            cv2.imshow("Client Camera Feed (Press Q to exit)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
