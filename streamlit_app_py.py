import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from deepface import DeepFace
import av
import cv2
 
st.set_page_config(page_title="Real-time Emotion & Spoof Detection", layout="centered")
st.title("Real-time Emotion Recognition with Spoof Detection")
 
class EmotionSpoofDetector(VideoTransformerBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
 
        try:
            face_objs = DeepFace.extract_faces(
                img_path=img,
                enforce_detection=False,
                detector_backend="opencv",
                align=True,
                anti_spoofing=True
            )
 
            for face_obj in face_objs:
                #x, y, w, h = face_obj['facial_area'].values()
                x = face_obj['facial_area']['x']
                y = face_obj['facial_area']['y']
                w = face_obj['facial_area']['w']
                h = face_obj['facial_area']['h']
                is_real = face_obj.get("is_real", False)
 
                # Draw bounding box
                color = (0, 255, 0) if is_real else (0, 0, 255)
                label = "Real" if is_real else "Spoof"
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
 
                # If real face, predict emotion
                if is_real:
                    analysis = DeepFace.analyze(
                        img[y:y + h, x:x + w],
                        actions=["emotion"],
                        enforce_detection=False
                    )
                    emotion = analysis[0]["dominant_emotion"]
                    cv2.putText(img, f"{emotion}", (x, y + h + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
 
        except Exception as e:
            cv2.putText(img, f"Error: {str(e)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
 
        return av.VideoFrame.from_ndarray(img, format="bgr24")
 
 
webrtc_streamer(
    key="emotion-spoof-stream",
    video_transformer_factory=EmotionSpoofDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
