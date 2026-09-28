# Acne Detection & Treatment Assistant

A Streamlit app that detects acne in an uploaded face photo using a fine-tuned YOLOv5 model and answers basic skincare questions from a JSON knowledge base.

> **Disclaimer:** This is a learning project, not a medical tool. It does not diagnose skin conditions. For persistent or severe acne, see a dermatologist.

## Features

- Upload a face photo (JPG or PNG)
- Detect acne spots and draw bounding boxes with confidence scores
- Adjustable confidence threshold in the sidebar
- Ask a treatment question and get advice from a local knowledge base

## How it works

1. **Detection:** the uploaded image is passed to a YOLOv5 model (Ultralytics `YOLO`) loaded from `best.pt`. Detections below the chosen confidence threshold are dropped, and the rest are drawn on the image with Pillow.
2. **Chatbot:** the user's question is lowercased and matched against the topics in `knowledge_base.json`. If a topic appears in the question, its stored advice is shown. This is a simple keyword lookup, not a language model.

Topics in the knowledge base: blackhead, whitehead, pimple, cystic acne, oily skin, dry skin, general acne care, diet, sun protection.

## Model

| Item | Value |
|---|---|
| Base model | `yolov5su.pt` (Ultralytics YOLOv5u, pretrained) |
| Fine-tuning | 50 epochs, image size 640, batch size 16 |
| Framework | Ultralytics 8.3.185 |
| Classes | `acne`, `fore` <!-- TODO: say what "fore" means, or delete this row --> |
| Dataset | <!-- TODO: dataset name and link --> |

Validation results of the saved `best.pt`:

| Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|
| 0.23 | 0.21 | 0.18 | 0.047 |

## Limitations

- **Modest accuracy.** The model finds only some acne spots and produces false detections. Acne lesions are small and low-contrast, and the model was trained for a short time.
- **Upload only.** There is no webcam or live video input.
- **Class labels.** The app labels every detection as "acne", even though the model has two classes.
- **Keyword chatbot.** It only answers questions that contain one of the nine topics. Other questions return "No advice found". It does not use the detection results.
- Results depend on lighting and image quality.

## Possible improvements

- Train longer on more and cleaner data, with a larger model and higher-resolution input
- Use the class names from the model instead of a fixed label
- Replace the keyword lookup with retrieval (TF-IDF or embeddings) so the chatbot finds relevant advice for any question, and optionally phrase answers with a language model
- Use the number of detected spots to tailor the advice
- Live webcam input with `streamlit-webrtc`

## Project structure

```
├── app.py                # Streamlit app
├── best.pt               # fine-tuned YOLOv5 weights
├── knowledge_base.json   # treatment advice used by the chatbot
├── requirements.txt
└── README.md
```

The training dataset is not included.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Keep `best.pt` and `knowledge_base.json` in the same folder as `app.py`.

## Tools

Python, Streamlit, Ultralytics YOLO, PyTorch, Pillow, NumPy
