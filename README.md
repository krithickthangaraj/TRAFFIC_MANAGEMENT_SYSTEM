# Traffic Management System

A computer vision–based traffic monitoring and management system that detects vehicles, manages single-lane priorities, and provides an interactive dashboard for monitoring real-time data.

## 📸 Screenshots

### Dashboard View
<img width="1438" height="891" alt="Detection Screenshot" src="https://github.com/user-attachments/assets/738945b6-3ed6-4f19-bc0c-b94d7e898ed9" />

### Detection View
<img width="1014" height="808" alt="Screenshot 2025-08-10 at 9 57 07 AM" src="https://github.com/user-attachments/assets/79356c9f-671f-46f7-acbe-8240c01fd6aa" />

## 🚀 Features
- **Vehicle Detection** using YOLOv3  
- **Traffic Priority Management** for single-lane scenarios  
- **Web Dashboard** to monitor vehicle counts and alerts  
- **Customizable Configuration** for camera inputs and detection zones  

## 🛠 Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/krithickthangaraj/TRAFFIC_MANAGEMENT_SYSTEM.git
    cd TRAFFIC_MANAGEMENT_SYSTEM
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # Mac/Linux
    venv\Scripts\activate      # Windows
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Add YOLO model weights (`yolov3.weights`) to the project folder (not included due to file size limits).

## ▶ Usage

Run the application:
```bash
python app.py
