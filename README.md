# 🚀 Detect_Disease

A comprehensive project for disease detection using machine learning models, featuring a robust API, utilities, and research notebooks.

---

## 📁 Project Structure

| Folder      | Description                                                        |
|-------------|--------------------------------------------------------------------|
| `API`       | Backend API for model inference and management (no large models)   |
| `Spam`      | Additional FastAPI-based backend utilities and scripts             |
| `Sugarcane` | Jupyter notebooks and model files for sugarcane disease detection |
| `.idea`     | Project configuration files for IDE (e.g., PyCharm)                |

---

## 🛠️ Dependencies & Compatible Versions

### API ([requirements.txt](API/App/requirements.txt))
- `tensorflow` >=2.16
- `fastapi` >=0.110, <0.120
- `uvicorn` >=0.29, <0.30
- `numpy` >=1.22, <1.26
- `pillow` >=9.0, <11.0
- `python-multipart` >=0.0.6

### Spam ([requirements_B.txt](Spam/requirements_B.txt), [requirements_P.txt](Spam/requirements_P.txt))
- `tensorflow` ==2.18.0
- `fastapi` >=0.110, <0.120
- `uvicorn` >=0.29, <0.30
- `numpy` ==1.26.0
- `pillow` >=9.0, <11.0
- `python-multipart` >=0.0.6

---

## 📦 Model Files

> **Note:** The large model files are not included in this repository due to size restrictions. You can download the required model files from the following Google Drive link:
>
> **[⬇️ Download Model Files from Google Drive](https://drive.google.com/drive/folders/1WS2QldEoiLzjIfPSkzsg6ilqI_SWzfzH?usp=drive_link)**

---

## 📝 Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Soumik2426/Detect_Disease.git
   ```
2. **Install dependencies:**
   - For API: `pip install -r API/App/requirements.txt`
   - For Spam: `pip install -r Spam/requirements_B.txt` or `Spam/requirements_P.txt`
3. **Download model files** from the Google Drive link above and place them in the appropriate directory as required by the code.
4. **Run the API or scripts** as described in the respective folder's documentation or code comments.

---

## 📚 Documentation & Support
- For detailed setup and usage, refer to the documentation in each folder or the code comments.
- For issues, please open an issue on the [GitHub repository](https://github.com/Soumik2426/Detect_Disease).

---

> Made with ❤️ by Soumik2426
