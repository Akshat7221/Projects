
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd, os
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = Flask(__name__, static_folder='static', static_url_path='/static')

def load_applicant_df(use='sample'):
    # use uploaded if requested
    if use == 'upload' and os.path.exists(os.path.join(UPLOAD_FOLDER,'uploaded_applicants.csv')):
        df = pd.read_csv(os.path.join(UPLOAD_FOLDER,'uploaded_applicants.csv'))
    else:
        # read original excel copy included
        df = pd.read_excel(r'/mnt/data/Applicants - Dayananda Sagar.xlsx')
    return df

def compute_simple_kpis(df):
    k = {}
    k['total_applicants'] = int(len(df))
    # try to map CGPA
    cgpa_cols = [c for c in df.columns if 'CGPA' in c.upper() or 'PERCENTAGE IN GRADUATION' in c.upper() or 'PERCENTAGE IN' in c.upper()]
    if cgpa_cols:
        try:
            k['avg_cgpa'] = round(pd.to_numeric(df[cgpa_cols[0]], errors='coerce').mean(),2)
        except:
            k['avg_cgpa'] = None
    else:
        k['avg_cgpa'] = None
    # Experience mapping
    exp_cols = [c for c in df.columns if 'EXPERIENCE' in c.upper()]
    if exp_cols:
        try:
            k['avg_experience'] = round(pd.to_numeric(df[exp_cols[0]], errors='coerce').mean(),2)
        except:
            k['avg_experience'] = None
    else:
        k['avg_experience'] = None
    # status counts
    status_col = None
    for c in df.columns:
        if c.strip().lower() == 'status':
            status_col = c
            break
    if status_col:
        k['status_counts'] = df[status_col].fillna('Unknown').value_counts().to_dict()
    else:
        k['status_counts'] = {}
    # branch mapping
    branch_col = None
    for c in df.columns:
        if c.strip().lower() == 'branch':
            branch_col = c
            break
    if branch_col:
        k['branch_counts'] = df[branch_col].fillna('Unknown').value_counts().to_dict()
    else:
        k['branch_counts'] = {}
    return k

def compute_advanced_kpis(df):
    k = {}
    k['total_applicants'] = int(len(df))
    # Gender pie
    gender_col = None
    for c in df.columns:
        if 'GENDER' in c.upper():
            gender_col = c; break
    if gender_col:
        k['gender_counts'] = df[gender_col].fillna('Unknown').value_counts().to_dict()
    else:
        k['gender_counts'] = {}
    # Course distribution
    course_col = None
    for c in df.columns:
        if 'COURSE' in c.upper():
            course_col = c; break
    if course_col:
        k['course_counts'] = df[course_col].fillna('Unknown').value_counts().to_dict()
    else:
        k['course_counts'] = {}
    # Specialization word frequencies
    spec_col = None
    for c in df.columns:
        if 'SPECIALIZATION' in c.upper() or 'SPECIAL' in c.upper():
            spec_col = c; break
    if spec_col:
        specs = df[spec_col].dropna().astype(str)
        freqs = specs.value_counts().to_dict()
        k['specialization_freq'] = freqs
    else:
        k['specialization_freq'] = {}
    # Graduation percentage distribution - find a numeric grad % column
    grad_cols = [c for c in df.columns if 'GRADUATION' in c.upper() or 'GRADUATION' in c.upper() or 'PERCENTAGE IN GRADUATION' in c.upper()]
    if grad_cols:
        arr = pd.to_numeric(df[grad_cols[0]], errors='coerce').dropna().tolist()
        k['graduation_percentages'] = arr
    else:
        # try other percentage columns
        pct_cols = [c for c in df.columns if 'PERCENTAGE' in c.upper() and 'GRADUATION' not in c.upper()]
        if pct_cols:
            arr = pd.to_numeric(df[pct_cols[0]], errors='coerce').dropna().tolist()
            k['graduation_percentages'] = arr
        else:
            k['graduation_percentages'] = []
    # Backlogs KPI (show only number of Yes)
    backlog_col = None
    for c in df.columns:
        if 'BACKLOG' in c.upper():
            backlog_col = c; break
    if backlog_col:
        yes_count = df[backlog_col].astype(str).str.contains('yes', case=False, na=False).sum()
        k['backlog_yes_count'] = int(yes_count)
    else:
        k['backlog_yes_count'] = 0
    # include sample table
    k['sample_table'] = df.head(200).to_dict(orient='records')
    return k

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/applicants-simple')
def simple_page():
    return send_from_directory('.', 'applicants_simple.html')

@app.route('/applicants-advanced')
def advanced_page():
    return send_from_directory('.', 'applicants_advanced.html')

@app.route('/api/applicants/simple', methods=['GET'])
def api_applicants_simple():
    use = request.args.get('use', 'sample')
    df = load_applicant_df(use=use)
    k = compute_simple_kpis(df)
    return jsonify({'kpis': k})

@app.route('/api/applicants/advanced', methods=['GET'])
def api_applicants_advanced():
    use = request.args.get('use', 'sample')
    df = load_applicant_df(use=use)
    k = compute_advanced_kpis(df)
    return jsonify(k)

@app.route('/api/upload/applicants', methods=['POST'])
def upload_applicants():
    if 'file' not in request.files:
        return jsonify({'error':'no file'}),400
    f = request.files['file']
    fname = f.filename.lower()
    if fname.endswith('.csv'):
        df = pd.read_csv(f)
    elif fname.endswith(('.xls','.xlsx')):
        df = pd.read_excel(f)
    else:
        return jsonify({'error':'unsupported'}),400
    save_path = os.path.join(UPLOAD_FOLDER, 'uploaded_applicants.csv')
    df.to_csv(save_path, index=False)
    return jsonify({'status':'ok'})

@app.route('/resume')
def resume():
    return send_from_directory('.', 'CV_Akshat_Jain.pdf')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
