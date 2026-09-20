from flask import Flask, render_template, request, redirect, send_from_directory, url_for
import csv
import io
import os
import logging
import datetime
from html import escape
from routes.projects import projects_bp

# `resend` is optional: if it isn't installed the site still works and
# messages are only saved to the CSV file.
try:
    import resend
except ImportError:
    resend = None

app = Flask(__name__)

# Register the new Blueprint
app.register_blueprint(projects_bp)

# ---------------------------------------------------------------------------
# Logging (only for this app's own messages, keeps Flask's request log clean)
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(_handler)
logger.propagate = False

# ---------------------------------------------------------------------------
# Email configuration (Resend)
# The API key must come from an environment variable. Never paste it in code.
# ---------------------------------------------------------------------------
RESEND_API_KEY = os.environ.get('RESEND_API_KEY')
NOTIFY_EMAIL = os.environ.get('NOTIFY_EMAIL', '213038045@student.presidency.edu.bd')

if resend and RESEND_API_KEY:
    resend.api_key = RESEND_API_KEY
    EMAIL_ENABLED = True
else:
    EMAIL_ENABLED = False
    logger.warning('Email is disabled (resend not installed or RESEND_API_KEY not set).')

# ---------------------------------------------------------------------------
# CSV configuration
# The CSV is a best-effort local copy. Even if the server cannot write files
# (some hosts are read-only), the email still includes a CSV attachment.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CSV_FILE = os.path.join(DATA_DIR, 'contact_messages.csv')
CSV_HEADER = ['time', 'name', 'email', 'message']

try:
    os.makedirs(DATA_DIR, exist_ok=True)
except OSError:
    logger.warning('Could not create the data folder; the CSV will only be sent by email.')


def safe_cell(value):
    # Stops Excel from running text like "=1+1" as a formula
    return "'" + value if value[:1] in ('=', '+', '-', '@') else value


def make_row(name, email, message):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return [now, safe_cell(name), safe_cell(email), safe_cell(message)]


# Function to save form data to CSV
def save_to_csv(row):
    is_new_file = not os.path.exists(CSV_FILE)
    # utf-8-sig so Excel shows Bangla and other non-English text correctly
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        if is_new_file:
            writer.writerow(CSV_HEADER)
        writer.writerow(row)


def build_csv_attachment(row, saved_to_disk):
    """Attach the full CSV file if it was saved, otherwise a one-row CSV."""
    data = None
    if saved_to_disk:
        try:
            with open(CSV_FILE, 'rb') as file:
                data = file.read()
        except OSError:
            data = None
    if data is None:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(CSV_HEADER)
        writer.writerow(row)
        data = ('\ufeff' + buffer.getvalue()).encode('utf-8')
    return {"filename": "contact_messages.csv", "content": list(data)}


# Function to email the message (with the CSV attached) to you
def send_notification(name, email, message, row, saved_to_disk):
    if not EMAIL_ENABLED:
        return
    try:
        resend.Emails.send({
            "from": "Website Contact <onboarding@resend.dev>",
            "to": [NOTIFY_EMAIL],
            "reply_to": email,
            "subject": f"New Contact Message from {name}"[:200],
            "html": (
                f"<p><strong>Name:</strong> {escape(name)}</p>"
                f"<p><strong>Email:</strong> {escape(email)}</p>"
                f"<p><strong>Message:</strong> {escape(message)}</p>"
            ),
            "attachments": [build_csv_attachment(row, saved_to_disk)],
        })
    except Exception:
        logger.exception('Email failed to send')


# Sample blog posts (can be replaced with dynamic data)
blog_posts = [
    {
        'title': 'Introduction to Flask',
        'content': 'Flask is a lightweight web framework for Python.',
        'date': '2023-10-01'
    },
    {
        'title': 'Web Scraping with BeautifulSoup',
        'content': 'Learn how to extract data from websites using Python.',
        'date': '2023-10-05'
    },
    {
        'title': 'Bootstrap for Beginners',
        'content': 'Create responsive websites with Bootstrap.',
        'date': '2023-10-10'
    }
]

@app.route('/')
def home():
    return render_template('home/index.html')

@app.route('/about')
def about():
    return render_template('about/about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        message = request.form.get('message', '').strip()

        # If something is missing, send them back to the form
        if not (name and email and message):
            return redirect(url_for('contact'))

        row = make_row(name, email, message)

        # Save to the CSV file. A failure here must not stop the email.
        saved_to_disk = False
        try:
            save_to_csv(row)
            saved_to_disk = True
        except Exception:
            logger.exception('Could not write to CSV')

        # Email the message with the CSV attached
        send_notification(name, email, message, row, saved_to_disk)

        # Redirect to the homepage after submission
        return redirect(url_for('home'))
    return render_template('contract/contact.html')

@app.route('/blog')
def blog():
    return render_template('blog/blog.html', posts=blog_posts)

@app.route('/resume')
def resume():
    return render_template('resume/resume.html')

@app.route('/design')
def design():
    return render_template('design/design.html')

@app.route('/estimate')
def estimate():
    return render_template('estimate/estimate.html')
@app.route('/others')
def others():
    return render_template('others/others.html')

@app.route('/scientific_calculator')
def scientific_calculator():
    return render_template('others/scientific_calculator.html')

@app.route('/speed_calculator')
def speed_calculator():
    return render_template('others/speed_converter.html')

@app.route('/mass_unit')
def mass_unit():
    return render_template('others/mass_unit.html')

@app.route('/pressure_converter')
def pressure_converter():
    return render_template('others/pressure_convert.html')

@app.route('/unit_weight')
def unit_weight():
    return render_template('others/unit_weight.html')

@app.route('/marge_pdf_tool')
def marge_pdf_tool():
    return render_template('others/marge_pdf_tool.html')

@app.route('/split_pdf_tool')
def split_pdf_tool():
    return render_template('others/split_pdf_tool.html')

@app.route('/compress_pdf_tool')
def compress_pdf_tool():
    return render_template('others/compress_pdf_tool.html')

@app.route('/beam_column_estimator')
def beam_column_estimator():
    return render_template('estimate/beam_column_estimator.html')

@app.route('/slab_materials_estimator')
def slab_materials_estimator():
    return render_template('estimate/slab_materials.html')

@app.route('/brick_calculator')
def brick_calculator():
    return render_template('estimate/brick_calculator.html')

@app.route('/projects')
def projects():
    return render_template('design/projects/projects.html')

@app.route("/beam_design")
def beam_design():
    return render_template("design/projects/beam_design/beam_design.html")

@app.route("/converter")
def converter():
    return render_template("others/converter/converter.html")

@app.route("/pdf_converter")
def pdf_converter():
    return send_from_directory("templates/others/converter", "pdf_converter.html")

@app.route("/pdf_editor")
def pdf_editor():
    return send_from_directory("templates/others/converter", "pdf_editor.html")

@app.route("/word_to_pdf")
def word_to_pdf():
    return send_from_directory("templates/others/converter", "word_to_pdf.html")

@app.route("/extract_remove_pages")
def extract_remove_pages():
    return send_from_directory("templates/others/converter", "extract_remove.html")

@app.route("/add_pdf_page")
def add_pdf_page():
    return send_from_directory("templates/others/converter", "add_pages.html")

@app.route("/add_page_number")
def add_pdf_number():
    return send_from_directory("templates/others/converter", "add_page_number.html")

@app.route("/image_to_pdf")
def image_to_pdf():
    return send_from_directory("templates/others/converter", "image_to_pdf.html")

@app.route("/excel_to_pdf")
def excel_to_pdf():
    return send_from_directory("templates/others/converter", "excel_to_pdf.html")

@app.route("/pdf_to_excel")
def pdf_to_excel():
    return send_from_directory("templates/others/converter", "pdf_to_excel.html")

@app.route("/powerpoint_to_pdf")
def powerpoint_to_pdf():
    return send_from_directory("templates/others/converter", "powerpoint_to_pdf.html")

@app.route("/pdf_to_powerpoint")
def pdf_to_powerpoint():
    return send_from_directory("templates/others/converter", "pdf_to_powerpoint.html")

@app.route("/powerpoint_page")
def powerpoint_page():
    return send_from_directory("templates/others/converter", "powerpoint_page.html")




@app.route("/two_beam_design")
def two_beam_design():
    return render_template("design/projects/beam_design/two_span_beam.html")

@app.route("/double_reinforced_beam_design")
def double_reinforced_beam_design():
    return render_template("design/projects/beam_design/doubly_reinforced.html")

@app.route("/t_beam_design")
def t_beam_design():
    return render_template("design/projects/beam_design/t_beam.html")

@app.route("/capacity_of_tbeam")
def capacity_of_tbeam():
    return render_template("design/projects/beam_design/capacity_of_t_beam.html")

@app.route("/stirrup_design")
def stirrup_design():
    return render_template("design/projects/beam_design/rebar_spacing_beam.html")

@app.route("/staircase_design")
def staircase_design():
    return render_template("design/projects/beam_design/stair_design.html")

@app.route("/column_design")
def column_design():
    return render_template("design/column_design.html")


@app.route("/bridge_design")
def bridge_design():
    return render_template("design/projects/bridge_design.html")

@app.route("/slab_design")
def slab_design():
    return render_template("design/slab_design.html")


@app.route("/projects/footing_design")
def footing_design():
    return render_template("design/projects/footing_design.html")

@app.route("/isolated_footing_design")
def isolated_footing_design():
    return render_template("design/projects/isolated footing design.html")

@app.route("/mat or raft_design")
def mat_or_raft_design():
    return render_template("design/projects/mat or raft_design.html")

@app.route("/telicommunicate_pile&pile_cap")
def telecommunication_pile_and_pile_cap():
    return render_template("design/projects/telicommunicate_pile&pile_cap.html")

@app.route("/multistory_building_design")
def multistory_building_design():
    return render_template("design/projects/multistory_pile_design.html")


@app.route("/pile_cap_design")
def pile_cap_design():
    return render_template("design/projects/pile_cap_design_tool.html")

@app.route("/pavement_design")
def pavement_design():
    return render_template("design/projects/pavement_design_tool.html")

@app.route("/rigid_pavement_design")
def rigid_pavement_design():
    return render_template("design/projects/rigid_pavement_design_tool.html")

@app.route("/marshall_mix_design")
def marshall_mix_design():
    return render_template("design/projects/marshall_mix_design_tool.html")

@app.route("/telecommucate_raft_design")
def telecommunication_raft_design():
    return render_template("design/projects/telecom_raft_foundation_tool.html")

@app.route("/masonry_building")
def masonry_building():
    return render_template("design/projects/brick_wall.html")

@app.route("/Frame_design")
def Frame_design():
    return render_template("design/projects/3D_Frame_design.html")

@app.route("/underground_overhead")
def underground_overhead():
    return render_template("design/projects/underground_overhead/underground_overhead.html")

@app.route("/underground_slab")
def underground_slab():
    return render_template("design/projects/underground_overhead/underground_slab.html")

@app.route("/underground_wall")
def underground_wall():
    return render_template("design/projects/underground_overhead/underground_wall.html")

@app.route("/overhead_slab")
def overhead_slab():
    return render_template("design/projects/underground_overhead/overhead_slab.html")

@app.route("/overhead_wall")
def overhead_wall():
    return render_template("design/projects/underground_overhead/overhead_wall.html")

@app.route("/underground_cover_slab")
def underground_cover_slab():
    return render_template("design/projects/underground_overhead/underground_cover_slab.html")

@app.route("/steel_design")
def steel_design():
    return render_template("design/steel_structure/steel_structure.html")

@app.route("/angle_design")
def angle_design():
    return render_template("design/steel_structure/angle_design.html")


@app.route("/w_section")
def w_section():
    return render_template("design/steel_structure/compression_w_section.html")

@app.route("/L_section")
def L_section():
    return render_template("design/steel_structure/compression_L_section.html")

@app.route("/bending_capacity")
def bending_capacity():
    return render_template("design/steel_structure/bending_capacity.html")
@app.route("/beam_column")
def beam_column():
    return render_template("design/steel_structure/beam_column.html")

@app.route("/base_plate_design")
def base_plate_design():
    return render_template("design/steel_structure/base_plate.html")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
