# UNIGUARD - Uniform Compliance System

> **Scan, Detect, and Stay Compliant**

**Developed with @Akash-GHB**

UniGuard is an enterprise-ready, premium and a powered security desktop application that helps automate student uniform compliance checks and biometric attendance logging in real-time. Uniguard is a Compliance System specific to Sathyabama institute of Science and Technology, Chennai.

**I’ve built an AI-powered compliance tool that automates uniform auditing and attendance tracking across campus entry gates. The tool automatically detects dress code compliance (like uniform shirts, ID badges, correct shoes), verifies student identities via facial recognition, and logs attendance details.
It solves a real problem: Gatekeepers manually inspect thousands of students daily, causing entry bottlenecks, human oversight, and inconsistent tracking.
My tool becomes a smart “gate sentinel” that scans, verifies, logs, and creates a real-time, hands-free compliance audit trail.**

**Unlike traditional gate checks that rely on slow, manual inspections, UniGuard is built around real-time automation, biometric validation, and frictionless compliance**.


---

## Problem vs Objective

<img width="1530" height="665" alt="Image" src="https://github.com/user-attachments/assets/31f04a93-cd82-4f45-b683-e6657d122e92" />


---

## Uni-Guard Interface

<img width="1600" height="999" alt="Image" src="https://github.com/user-attachments/assets/702f131b-dec2-48c6-bef3-ba15b5846d6c" />

<img width="1600" height="999" alt="Image" src="https://github.com/user-attachments/assets/6c3dd534-4e65-4700-92a2-7d57cbe083f5" />

<img width="1600" height="999" alt="Image" src="https://github.com/user-attachments/assets/c7548bf9-0915-4acd-a1e8-b5c9e37b92fc" />

<img width="1600" height="999" alt="Image" src="https://github.com/user-attachments/assets/85bb5c13-c353-44b3-b859-212c3c490c1c" />

---

## ✨ Core Idea

Most gate security processes rely on slow, manual eyesight checks… and inevitably miss infractions.

UniGuard flips this behavior by:

* Automating **attire compliance checks**
* Matching student identities using **zero-friction biometrics**
* Providing instant, **digital audit logs**
  
Think of it as a smart gate sentinel rather than a passive camera recorder.

---

## 🔑 Key Features

### Attire Verification

* Scan uniforms, ID badges, and footwear automatically using custom YOLO detectors
* Configurable compliance target tags (e.g., batch-specific tags)

### Instant Biometric Recognition

* Matches passing faces with database profiles on the fly via InsightFace
* Quick registration via a simple profile image drop (student_photos/)

### Live Database Explorer

* Track daily compliance logs and weekly matrices in real-time 
* Interactive cell edits, self-correction logging, and CSV exports

### Privacy First by Design

* Processes all frames locally on-edge without uploading biometric data to external servers
* Cached student database profiles for immediate connection disposal

---

## Flow Diagram

<img width="1512" height="572" alt="Image" src="https://github.com/user-attachments/assets/70d071a4-7cb7-4fd2-8970-b5a750c15ece" />


---

## 🧠 Philosophy (Why SaveStack?)

* ❌ Not a passive CCTV DVR recorder
* ❌ Not a slow, bottlenecked manual paper register
* ✅ A **smart compliance monitor**
* ✅ A **hands-free security assistant**

UniGuard helps you maintain standards at the gate — without slowing down entry.

---

## How does UNIGUARD work?


**Face Embedding is analyzed - System identifies student as "Shreyas S" (Me) - Scans my body for ID Card - Finds ID**


<img width="3200" height="1968" alt="Image" src="https://github.com/user-attachments/assets/46acc16f-d64b-4d3a-a90e-69be09876e3d" />


**Marks Student (Me) as Present for the Day**


<img width="1071" height="301" alt="Image" src="https://github.com/user-attachments/assets/6d162a67-b664-42e6-a6c8-73d50e608d06" />

**Changes to SQLite Database reflect inside Uniguard UI**

<img width="1600" height="539" alt="Image" src="https://github.com/user-attachments/assets/5b5813d2-4142-4d40-bc64-88fe700036d5" />

---

## 🛠 Tech Stack

### Vision and Core

* **Object Detection**: YOLOv8 (Ultralytics PyTorch / ONNX)
* **Face Recognition**: InsightFace (ONNX Runtime / CPU & GPU Execution)
* **Image Processing**: OpenCV (Multithreaded Frame Decoders)

### UI and Presentation

* **Application Framework**: PyQt6 (Python Qt6 bindings)
* **Interactive Charts**: PyQtGraph (Hardware-accelerated analytics plotting)
* **System Resource Monitoring**: Psutil

### Storage

* **Local Database**: SQLite3 (Optimized with WITHOUT ROWID storage tables)
* **Reference Cache**: NumPy (embeddings.npz vector matrix storage))

---

## 🧩 Architecture Overview

```text
Camera Stream (Webcam/IP) → OpenCV Frame Grabber
                          → YOLOv8 (Dress Code Compliance Checks)
                          → InsightFace (Face Embedding Matching)
                          → Database Manager (SQLite transactional log)
                          → PyQt6 Security Dashboard UI (Analytics / Tables)
```

---

## 🔄 Attendance and Verification Logic

1. Video frame is captured from the input stream.
2. YOLOv8 detects the presence of uniforms, ID badges, and footwear compliance tags.
3. InsightFace extracts facial embeddings and compares them with registered references.
4. If a student is identified, their attendance is logged:
5. Status is marked as PRESENT if all compliance tags are satisfied.
6. Status is marked as VIOLATION with details if any required item is missing.
7. Self-Correction Logic: Upgrades status from VIOLATION to PRESENT if the student corrects their attire and scans again on the same day.
8. Local logs are synchronized with the weekly attendance metrics grid.

---

## 🧪 Project Status

* ✅ Real-time YOLO uniform/shoes/badge detection
* ✅ InsightFace biometric extraction and auto-embedding generation
* ✅ SQLite Database schema auto-migrations (WITHOUT ROWID)
* ✅ Dynamic self-correction and priority logging logic
* 🚧 Multi-camera RTSP streaming support
* 🚧 Cloud synchronization engine (Central PostgreSQL database backend
* 🚧 On field deployment post Model Optimization and Hardware Optimization
* 🚧 Exterprise-Grade security with RBAC (Role Based Access Control)
* 🚧 Notification Pipelines (Automated)

---

## 🧭 Future Ideas

* Dynamic email/SMS notification alerts (Twilio / Resend integration)
* Integration with popular Student Information Systems (SIS)
* Edge device deployment templates (e.g., Nvidia Jetson / Intel OpenVINO)
* Active Directory/SSO integration for admin panel controls

---

## 🧑‍💻 Author

**Shreyas S**
Student Developer | Web • Cloud • AI

Built with curiosity, frustration with bookmarks, and a love for clean systems.

---

## 📄 License

MIT License

Copyright (c) 2026 Shreyas S

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

> *Detect infractions. Verify identities. Automate gatekeeping.*
