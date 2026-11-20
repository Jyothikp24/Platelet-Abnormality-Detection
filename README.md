\# Platelet Detection using YOLO



A final year project that detects and counts platelets in blood smear images using YOLO, trained on the BCCD (Blood Cell Count and Detection) dataset.



\---



\## Overview



Manual blood cell counting is slow and prone to human error. This project uses a YOLO object detection model to automatically locate platelets in microscopic images, making the counting process faster and more consistent.



\## Features



\- Detects platelets in blood smear images

\- Trained on the public BCCD dataset

\- Jupyter notebooks for dataset preparation and training

\- Easy to extend to other blood cell types (RBC, WBC)



\## Tech Stack



| Category | Tools |

|----------|-------|

| Language | Python |

| Model | YOLO (Ultralytics) |

| Environment | Jupyter Notebook |

| Image processing | OpenCV |



\## Project Structure



```

Platelets/

├── bccd\_yolo/      # Dataset preparation notebooks

└── runs/           # Training outputs (not included in repo)

```



\## Getting Started



\### 1. Clone the repository



```bash

git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git

cd YOUR-REPO

```



\### 2. Install dependencies



```bash

pip install ultralytics opencv-python jupyter

```



\### 3. Prepare the dataset



The BCCD dataset is not included in this repository. Download it from its original source and place it in the folder expected by the notebooks.



\### 4. Run the notebooks



```bash

jupyter notebook

```



Open the notebooks in `Platelets/bccd\_yolo/` and run the cells in order.



\## Results



| Metric | Value |

|--------|-------|

| Precision | \_add value\_ |

| Recall | \_add value\_ |

| mAP@0.5 | \_add value\_ |



\_Add sample detection images here.\_



\## Author



\*\*Your Name\*\*

Your College, Your Department

Batch of 2026

