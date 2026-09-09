from flask import Flask, render_template, request, redirect, url_for
import csv
import os
from routes.projects import projects_bp
from flask_mail import Mail, Message

app = Flask(__name__)

# Register the new Blueprint
app.register_blueprint(projects_bp)

# Mail configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')

mail = Mail(app)

# Ensure the 'data' directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# Path to the CSV file
CSV_FILE = 'data/contact_messages.csv'

# Function to save form data to CSV
def save_to_csv(name, email, message):
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, email, message])

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
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        
        # Save the form data to CSV (kept as backup)
        save_to_csv(name, email, message)

        # Send an email notification to yourself
        try:
            msg = Message(
                subject=f'New Contact Message from {name}',
                sender=app.config['MAIL_USERNAME'],
                recipients=[app.config['MAIL_USERNAME']],
                body=f'Name: {name}\nEmail: {email}\nMessage: {message}'
            )
            mail.send(msg)
        except Exception as e:
            print(f'Email failed to send: {e}')
        
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

#if __name__ == '__main__':
#    app.run(debug=True)
