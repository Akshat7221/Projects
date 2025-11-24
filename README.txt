
Combined Applicants Dashboards (v2)
- /applicants-simple : Simple dashboard (Name, Branch, CGPA, Experience, Status)
- /applicants-advanced : Advanced dashboard (uses full uploaded Excel)
Upload CSV/XLSX via the upload button on each page. The app will save uploaded data to uploads/uploaded_applicants.csv
Run:
pip install flask pandas openpyxl
python app.py
Open http://127.0.0.1:5000/applicants-simple or /applicants-advanced
